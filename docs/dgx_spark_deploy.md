# QwenTTS on NVIDIA DGX Spark (GB10 Blackwell, ARM64)

Deployment runbook for a single-node DGX Spark host running QwenTTS API in a Docker container,
based on the NGC PyTorch image and managed by `systemd`.

This document captures **what was done, what broke, and how it was worked around** during the
first deployment to the `gx10-1277` box (Ubuntu 24.04, GB10 Blackwell, CUDA 13.0). It is
NOT a clean spec; it is a runbook so the next person doesn't re-hit the same six wall.

## Why this is different from Vast.ai

| | Vast.ai recipe | DGX Spark |
|---|---|---|
| Arch | x86_64 | aarch64 (Grace ARM) |
| GPU | A100/H100 (sm_80/90) | GB10 Blackwell (sm_121) |
| CUDA | 12.1 | 13.0 |
| torch | 2.2.2+cu121 (PyPI wheel) | 2.8.0a0+...nv25.05 (NGC custom alpha) |
| torchaudio | works | ABI-incompatible with NGC torch alpha |
| faster-whisper GPU | works | ctranslate2 wheel for arm64 is CPU-only |
| Process lifecycle | interruptible, recreate-from-clean | persistent box, share GPU with other tenants |
| Worker queue | `task_worker` polling rixtrema task_api | NOT yet wired to prod (see Cutover) |

Two patches were made for cross-compatibility: see "Code patches" below.

## Hardware / OS audit

```
uname -m           # aarch64
lsb_release -a     # Ubuntu 24.04.4 LTS (noble)
nvidia-smi         # NVIDIA GB10, driver 580.x, CUDA 13.0
                   # memory.total/used/free → N/A (unified memory)
```

The 128 GiB LPDDR5X is **unified** — the same physical RAM serves CPU and GPU. There is no
discrete VRAM. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
DOES show per-process GPU usage; `--query-gpu=memory.*` returns `N/A`.

## Filesystem layout

```
/opt/qwentts/                  ← git clone of QwenTTS repo
  data/queue.db                ← API sqlite queue
  logs/                        ← uvicorn logs, task_worker logs
  tmp/
  qwentts.env                  ← copy of /etc/qwentts.env (container-visible via volume mount)
/etc/qwentts.env               ← master env file (chmod 600, root:root)
/etc/qwentts.env.base          ← non-secret part of env (re-mergeable)
/etc/systemd/system/qwentts.service
/home/rixtrema/.cache/huggingface/  ← HF cache (Qwen3-TTS-1.7B-Base, Whisper large-v3)
~/qwentts-prep/                ← scratch venv for prefetch / one-off scripts
```

## Setup order (proven, no shortcuts)

### 1. Remote access (so AnyDesk stops being your lifeline)

The host has no static IP. SSH access is via the owner's Cloudflare Tunnel:

- Owner runs `cloudflared.exe service install` on DGX → exposes `ssh-dgx.irafiduciaryoptimizer.com`
- Owner adds you to Cloudflare Access policy for that hostname (email + OTP)
- On your laptop install `cloudflared` (`winget install Cloudflare.cloudflared`) and add to `~/.ssh/config`:
  ```
  Host dgx-via-cf
      HostName ssh-dgx.irafiduciaryoptimizer.com
      User rixtrema
      IdentityFile ~/.ssh/id_ed25519
      ProxyCommand "C:\Program Files (x86)\cloudflared\cloudflared.exe" access ssh --hostname %h
  ```
- Generate ed25519 keypair on the laptop, push public part to `~/.ssh/authorized_keys` on DGX.

**Important gotcha**: `tailscale up --ssh` enables Tailscale's own in-process SSH server that
**intercepts** port 22 before sshd sees it. If sshd handshake is `Connection closed by remote
host` before banner exchange, that's Tailscale SSH preempting. Disable with
`sudo tailscale set --ssh=false && sudo systemctl restart tailscaled`. Regular openssh-server
then takes over normally.

If your laptop has another VPN (e.g. Hiddify/Happ) hijacking the `100.x.y.z` CGNAT range,
Tailscale tunnel won't route to the DGX peer. Cloudflare Tunnel sidesteps this entirely.

### 2. Docker daemon

First obstacle was `dockerd` crashing on start with
`error initializing buildkit: error creating buildkit instance: invalid database`. Wipe
the corrupted buildkit cache (containers and images survive):

```bash
sudo systemctl stop docker.socket docker.service
sudo rm -rf /var/lib/docker/buildkit
sudo systemctl reset-failed docker.service
sudo systemctl start docker.service
sudo systemctl is-active docker   # → active
```

By default rixtrema is NOT in the `docker` group — every command needs `sudo docker ...`. Ask
the host owner to add: `sudo usermod -aG docker rixtrema` + re-login.

### 3. Pre-fetch HF models (defends against IPv6 hangs at runtime)

Outbound IPv6 from this host is broken (CGNAT / ISP). `huggingface_hub` resolves
huggingface.co AAAA records first and stalls in `SYN-SENT` for the full Linux TCP retry
window (~3 min) before failing back to IPv4. To avoid this in production, **pre-fetch the
models once with IPv4 forced**:

```bash
python3 -m venv ~/qwentts-prep
source ~/qwentts-prep/bin/activate
pip install -U pip huggingface_hub

# Qwen3-TTS Base (~4.3 GB; this is the model server/tts.py loads)
nohup python -c "
import socket
_o = socket.getaddrinfo
def v4(*a,**k): return [r for r in _o(*a,**k) if r[0]==socket.AF_INET]
socket.getaddrinfo = v4
from huggingface_hub import snapshot_download
print('MODEL_PATH=', snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base'))
" > ~/hf-base-download.log 2>&1 &
```

The Whisper `large-v3` model is downloaded lazily on first ASR call (~2.9 GB) — once
present in cache, runtime is fine.

After both are cached, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the env so
`snapshot_download` never makes a network call at all (see env file below).

### 4. Pull the NGC PyTorch container

```bash
nohup sudo docker pull nvcr.io/nvidia/pytorch:25.05-py3 > ~/ngc-pull.log 2>&1 &
# ~10 GB compressed, ~24 GB on disk after extract
```

Confirmed `25.05-py3` recognises GB10 sm_121 and can run a matmul:

```bash
sudo docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.05-py3 \
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# torch: 2.8.0a0+5228986c39.nv25.05 | NVIDIA GB10 | (12, 1)
```

The image warns `Detected NVIDIA GB10 GPU, which may not yet be supported in this version
of the container`. In practice all base ops (matmul, attention, Qwen3-TTS inference) work;
some specialised kernels (TransformerEngine fp8 etc.) might not — none of those are on
QwenTTS's path. If a particular kernel fails later, bump to a newer NGC release
(`25.10-py3` or beyond).

### 5. Install QwenTTS deps inside the container

Run a long-lived container with `sleep infinity`, install deps via `docker exec`, then
commit to a `qwentts:dev` image so the install survives any restart:

```bash
git clone https://github.com/greenbutton75/QwenTTS.git /opt/qwentts

sudo docker run -d \
  --name qwentts-dev \
  --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --net=host \
  -v /opt/qwentts:/opt/qwentts \
  -v /home/rixtrema/.cache/huggingface:/root/.cache/huggingface \
  -w /opt/qwentts \
  nvcr.io/nvidia/pytorch:25.05-py3 \
  sleep infinity

# Install in detached mode so SSH disconnect doesn't kill it.
# Log goes to a host-side path (volume mount) so it survives even container restart.
sudo docker exec -d qwentts-dev bash -c '
pip install fastapi==0.110.0 "uvicorn[standard]==0.27.1" python-multipart==0.0.9 \
  boto3==1.34.59 pydub==0.25.1 soundfile==0.12.1 einops==0.7.0 \
  librosa==0.10.1 setuptools==75.8.0 sox==1.4.1 \
  huggingface_hub==0.34.0 safetensors==0.4.3 tokenizers==0.22.2 \
  regex==2024.11.6 requests==2.31.0 > /opt/qwentts/pip-install.log 2>&1 && \
pip install --no-deps transformers==4.57.3 accelerate==1.12.0 >> /opt/qwentts/pip-install.log 2>&1 && \
pip install onnxruntime==1.17.3 >> /opt/qwentts/pip-install.log 2>&1 && \
pip install faster-whisper==1.0.3 kaldi-native-fbank==1.21.5 >> /opt/qwentts/pip-install.log 2>&1 && \
echo ALL_OK >> /opt/qwentts/pip-install.log'

# Wait, watch the log, then snapshot.
tail -f /opt/qwentts/pip-install.log
sudo docker commit qwentts-dev qwentts:dev
```

**Critical**: `--no-deps` on transformers/accelerate prevents pip from "upgrading" the
NGC torch 2.8 alpha to whatever PyPI thinks is compatible. The NGC torch is the only
build that knows about sm_121; replacing it kills GPU support.

### 6. Patches required for the NGC ARM stack

Two patches landed on `main` (commits `99051ef`, `fb40e2c`). Both preserve old behaviour
on Vast.ai and only activate on hosts where the original code path fails:

- **`qwen_tts/core/tokenizer_25hz/vq/speech_vq.py`** — try `torchaudio.compliance.kaldi`,
  fall back to `kaldi_native_fbank` (pure C++, no torch ABI dependency). torchaudio wheels
  on PyPI are not ABI-compatible with the NGC torch alpha; `kaldi_native_fbank` produces
  bit-equivalent fbank features so the downstream X-Vector ONNX encoder is unaffected.
- **`server/asr.py`** + **`server/config.py`** — new `ASR_DEVICE` env (`auto`/`cuda`/`cpu`).
  The arm64 PyPI wheel of `ctranslate2` is CPU-only; loading Whisper with `device="cuda"`
  crashes even though `torch.cuda.is_available()` is `True`. Set `ASR_DEVICE=cpu` on this
  host so Whisper falls back to int8 CPU inference (slower but works).

`kaldi-native-fbank==1.21.5` was added to `server/requirements.txt` so a clean install on
any host has the fallback available.

### 7. Build /etc/qwentts.env

The env has three parts: AWS/S3 creds, prod secrets (TASK_BASE_URL, USER_TOKEN, SYSTEM_TOKEN
— pulled from `s3://rixtrema-qwentts/secrets/qwentts.env`), and DGX-specific knobs.

Key DGX-specific values:

```
FINGERPRINT=dgx-spark-001          # unique identifier so task_api can route per-worker
ASR_DEVICE=cpu                     # ctranslate2 wheel here is CPU-only
HF_HUB_OFFLINE=1                   # never call HF Hub at runtime (caches already populated)
TRANSFORMERS_OFFLINE=1             # same intent, transformers side
SQLITE_PATH=/opt/qwentts/data/queue.db
LOG_DIR=/opt/qwentts/logs
```

Build helper (assembles secrets + base + dedups, sets 600 root:root):

```bash
sudo docker exec -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_REGION=us-east-1 \
  qwentts-dev python -c "
import boto3
data = boto3.client('s3').get_object(Bucket='rixtrema-qwentts', Key='secrets/qwentts.env')['Body'].read().decode().replace('\r\n','\n')
open('/opt/qwentts/qwentts.secrets.env','w').write(data if data.endswith('\n') else data+'\n')
"

sudo tee /etc/qwentts.env.base > /dev/null <<'EOF'
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=rixtrema-qwentts
S3_PREFIX=support
FINGERPRINT=dgx-spark-001
MODEL_SIZE=1.7B
LANGUAGE=English
QWEN_TTS_BASE_URL=http://127.0.0.1:8000
QWEN_TTS_HOST=0.0.0.0
QWEN_TTS_PORT=8000
SQLITE_PATH=/opt/qwentts/data/queue.db
LOG_DIR=/opt/qwentts/logs
TASK_WORKER_HEALTH_PORT=8010
TASK_WORKER_LOG_DIR=/opt/qwentts/logs
ASR_DEVICE=cpu
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
ADMIN_USER=admin
ADMIN_PASSWORD=<generate>
# ... plus the GREETING_*, OUTPUT_AUDIO_*, ENABLE_PHRASE_SPLICE_GROUPING flags
# matching the prod values in README.md / qwentts.env
EOF

sudo bash -c '{ grep -v "^FINGERPRINT=" /opt/qwentts/qwentts.secrets.env; grep -v "^TASK_BASE_URL=" /etc/qwentts.env.base; } > /etc/qwentts.env; chmod 600 /etc/qwentts.env; chown root:root /etc/qwentts.env'

# Dedup any accidental duplicate keys (last write wins via shell `source`, but visually noisy):
sudo python3 -c "
seen = set(); out = []
for line in open('/etc/qwentts.env'):
    k = line.split('=', 1)[0]
    if k not in seen: seen.add(k); out.append(line)
open('/etc/qwentts.env','w').write(''.join(out))
"
sudo chmod 600 /etc/qwentts.env

# Mirror to /opt/qwentts so the container (which doesn't bind-mount /etc) can source it.
sudo cp /etc/qwentts.env /opt/qwentts/qwentts.env
sudo chmod 600 /opt/qwentts/qwentts.env
```

Runtime dirs:

```bash
sudo mkdir -p /opt/qwentts/data /opt/qwentts/logs /opt/qwentts/tmp
sudo chown -R rixtrema:rixtrema /opt/qwentts/data /opt/qwentts/logs /opt/qwentts/tmp
```

### 8. systemd unit

`/etc/systemd/system/qwentts.service`:

```ini
[Unit]
Description=QwenTTS API (containerized)
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=15
ExecStartPre=-/usr/bin/docker stop qwentts
ExecStartPre=-/usr/bin/docker rm qwentts
ExecStart=/usr/bin/docker run --rm --name qwentts \
  --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --net=host \
  --env-file /etc/qwentts.env \
  -v /opt/qwentts:/opt/qwentts \
  -v /home/rixtrema/.cache/huggingface:/root/.cache/huggingface \
  -w /opt/qwentts \
  -e PYTHONPATH=/opt/qwentts \
  -e PYTHONUNBUFFERED=1 \
  qwentts:dev \
  python -u -m uvicorn server.app:app --host 0.0.0.0 --port 8000
ExecStop=/usr/bin/docker stop qwentts

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable qwentts
sudo systemctl start qwentts
sleep 25
curl -fsS http://127.0.0.1:8000/health && echo
```

### 9. Smoke test

Synchronous splice (no S3 writes, just returns WAV):

```bash
curl --max-time 600 -X POST http://127.0.0.1:8000/phrases/splice-test \
  -H 'Content-Type: application/json' \
  -d '{"support_id":"85159","voice_id":"<some-existing-voice>","greeting":"Hi Alex.","body":"This is a test from DGX Spark.","pause_ms":150,"crossfade_ms":40,"content_aware":true,"target_lufs":-16,"mode":"wav_splice"}' \
  -D /tmp/dgx-smoke.headers --output /tmp/dgx-smoke.wav

# First request: 30-60 s (model loads into 5.4 GB unified memory). Subsequent: ~10 s.
# Expected: ~200 KB WAV, X-Greeting-Similarity > 0.9, X-Greeting-Passed: true.
```

Async (full prod path: sqlite queue → internal worker → S3):

```bash
TS=$(date +%s)
curl -s -X POST http://127.0.0.1:8000/phrases -H 'Content-Type: application/json' -d \
  "{\"support_id\":\"85159\",\"voice_id\":\"<some-existing-voice>\",\"phrase_id\":\"dgx_e2e_${TS}_1\",\"text\":\"Hi Alex, this is Sarah from Rixtrema.\"}"
# Poll status; expect "done" within 30-60 s, public_url filled.
```

S3 layout produced by both:
- `s3://rixtrema-qwentts/support/{support_id}/phrases/{phrase_id}.wav`
- `s3://rixtrema-qwentts/support/{support_id}/phrases/{phrase_id}.json` (status, splice metadata)

## Surprises that ate hours (so you don't re-debug them)

1. **`Tailscale up --ssh` intercepts port 22.** sshd is silent; connection closes before
   banner. Fix: `sudo tailscale set --ssh=false && sudo systemctl restart tailscaled`.

2. **`docker run --rm -it ... bash` dies on SSH disconnect.** TTY HUP → bash exits → `--rm`
   removes the container → all your `pip install`s are gone. Use `docker run -d ... sleep
   infinity` + `docker exec -d` for installs. The `-d` on exec is important — it detaches
   inside the container, so SSH HUP doesn't propagate.

3. **Outbound IPv6 is broken on this network.** `huggingface_hub`, `pytorch.org`
   downloads, anything that resolves AAAA first stalls for ~3 min in SYN-SENT before
   falling back to IPv4. Workaround for downloads: monkey-patch `socket.getaddrinfo` to
   strip AF_INET6. Workaround for runtime: `HF_HUB_OFFLINE=1` so HF never reaches the
   network.

4. **`torchaudio` from PyPI is ABI-incompatible with NGC torch alpha.** Symptom:
   `OSError: undefined symbol: torch_library_impl` (PyPI built against newer torch) or
   `libcudart.so.13 not found` (built against newer CUDA). Don't waste time. Use
   `kaldi-native-fbank` for the one place we need kaldi fbank features (already patched).

5. **`ctranslate2` arm64 wheel is CPU-only.** Loading Whisper with `device="cuda"`
   crashes with `ValueError: This CTranslate2 package was not compiled with CUDA support`,
   even though `torch.cuda.is_available()` is True. Set `ASR_DEVICE=cpu`. Whisper
   large-v3 on Grace ARM CPU is ~0.5-1.5 RTF — slower than GPU but functional.

6. **Coexistence with `llama-server` (gpt-oss-120b) is uncertain.** On this box the
   neighbour holds ~64 GB unified memory. Even with ~45 GB free on paper, the NVRM
   `NV_ERR_NO_MEMORY` fired when loading our 5 GB Qwen model — GPU contexts need
   contiguous reservations and unified-memory fragmentation can make the math lie. We
   smoke-tested with `llama-server` stopped. Real 24/7 coexistence is **not yet
   verified**. The DGX side is `systemctl stop qwentts` / `systemctl start qwentts` to
   give/take the GPU; the llama side has its own start script.

## Cutover plan (not done yet — see Phase 9 todo)

To make DGX a prod worker that picks up real tasks from `rixtrema.net/api/async_task_manager`:

1. Write a second systemd unit `qwentts-worker.service` that runs the same container with
   command `python -m task_worker.main`. It polls task_api with our `USER_TOKEN`,
   `SYSTEM_TOKEN`, `FINGERPRINT=dgx-spark-001`.
2. **Open question**: does task_api dispatch by FINGERPRINT? If yes — DGX worker and Vast
   worker can run in parallel, splitting load by fingerprint identity. If no — only one
   worker can run at a time; we must stop Vast before starting DGX worker.
3. Coexistence with llama-server must be settled (see surprise #6).

Until those are resolved, DGX runs only the API service — useful for local smoke /
testing /admin dashboards but does not consume the prod queue.

## Operational follow-ups

- Rotate AWS keys (`AKIA322GJ6UJFW6EAGOZ`) — they appeared in setup chat, treat as
  compromised.
- Add `ffmpeg` inside the image (pydub warns; not blocking but cleaner): `apt-get install
  ffmpeg && docker commit`.
- Backup `/opt/qwentts/data/queue.db` to S3 hourly (cron) — protects against disk loss.
- Reboot test was skipped because the host has live tenants. To verify autostart works,
  schedule a maintenance window and confirm `systemctl is-active qwentts` after the
  reboot.
- External `/health` and `/admin` access: currently only via `ssh -L 8000:127.0.0.1:8000
  dgx-via-cf -N`. For production monitoring add a Cloudflare Tunnel HTTP route
  (`qwentts.example.com → http://127.0.0.1:8000`) with Access policy, owned by the
  cloudflared service that already runs on this box.

## Where to look when something breaks

| Symptom | Where |
|---|---|
| API up but `/phrases` hangs | `sudo py-spy dump --pid $(sudo lsof -ti :8000)` — almost certainly a network IO stall. Confirm `HF_HUB_OFFLINE=1`. |
| `NV_ERR_NO_MEMORY` in `dmesg` | Neighbour (`llama-server`) is holding the GPU. `sudo systemctl stop qwentts` until free. |
| `undefined symbol: torch_library_impl` | Someone reinstalled `torchaudio`. Re-run the kaldi_native_fbank fallback (already in code) and uninstall torchaudio. |
| `This CTranslate2 package was not compiled with CUDA` | `ASR_DEVICE` got reset. Add it back to `/etc/qwentts.env`. |
| `Connection closed by remote host` on `ssh dgx-via-cf` | `cloudflared` access cache expired or Tunnel down. `cloudflared access login ssh-dgx.irafiduciaryoptimizer.com` to refresh. |
| `tail uvicorn.out` empty after restart | Stdio buffered. systemd unit runs `python -u`; if you launched manually, also set `PYTHONUNBUFFERED=1`. |

## Verified versions (snapshot at first deploy)

- NGC image: `nvcr.io/nvidia/pytorch:25.05-py3`
- torch: `2.8.0a0+5228986c39.nv25.05`
- transformers: `4.57.3`
- accelerate: `1.12.0`
- numpy: `1.26.4`
- faster-whisper: `1.0.3` (CPU)
- ctranslate2: `4.7.2` (CPU)
- kaldi-native-fbank: `1.21.5`
- huggingface_hub: `0.34.0`
- py-spy: `0.4.2` (debug only)
- Qwen3-TTS Base: snapshot `fd4b254389122332181a7c3db7f27e918eec64e3`

## End-to-end measured timings (no neighbour load)

- Cold start (load Qwen3-TTS into GPU + Whisper into RAM): ~30-60 s
- Splice `/phrases/splice-test` after warmup: ~10 s
- Async `/phrases` after warmup: ~10-15 s (body_cache hit) / ~30-60 s (body_cache miss + body best-of-N)
- GPU memory held while serving: ~5.4 GiB (Qwen) + transient tensors
- CPU memory held while serving: ~3-5 GiB (Whisper + ONNX + buffers)
