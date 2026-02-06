
# Python API documentation for http://127.0.0.1:7860/
API Endpoints: 17

1. Install the Python client [docs](https://www.gradio.app/guides/getting-started-with-the-python-client) if you don't already have it installed. 

```bash
pip install gradio_client
```

2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data. 

### API Name: /generate_custom_voice
Description: Генерация с пресетами голосов (стриминг).

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	text="Привет! Это демонстрация системы синтеза речи Qwen3-TTS, портативная версия от канала Нейро-софт. Она поддерживает русский язык и множество других языков.",
	language="Русский",
	speaker="Вивиан (женский, английский)",
	instruct="Hello!!",
	model_size="1.7B",
	max_tokens=2048,
	temperature=0.7,
	top_p=0.9,
	api_name="/generate_custom_voice"
)
print(result)
```

Accepts 8 parameters:

text:
- Type: str
- Default: "Привет! Это демонстрация системы синтеза речи Qwen3-TTS, портативная версия от канала Нейро-софт. Она поддерживает русский язык и множество других языков."
- The input value that is provided in the Текст для синтеза Textbox component. 

language:
- Type: Literal['Авто (определить автоматически)', 'Русский', 'Английский', 'Китайский', 'Японский', 'Корейский', 'Французский', 'Немецкий', 'Испанский', 'Португальский', 'Итальянский']
- Default: "Русский"
- The input value that is provided in the Язык Dropdown component. 

speaker:
- Type: Literal['Эйден (мужской, английский)', 'Дилан (мужской, английский)', 'Эрик (мужской, английский)', 'Анна (женский, японский)', 'Райан (мужской, английский)', 'Серена (женский, английский)', 'Сохи (женский, корейский)', 'Дядя Фу (мужской, китайский)', 'Вивиан (женский, английский)']
- Default: "Вивиан (женский, английский)"
- The input value that is provided in the Голос Dropdown component. 

instruct:
- Type: str
- Required
- The input value that is provided in the Стиль (опционально) Textbox component. 

model_size:
- Type: Literal['0.6B', '1.7B']
- Default: "1.7B"
- The input value that is provided in the Размер модели Dropdown component. 

max_tokens:
- Type: float
- Default: 2048
- The input value that is provided in the Макс. токенов Slider component. 

temperature:
- Type: float
- Default: 0.7
- The input value that is provided in the Температура Slider component. 

top_p:
- Type: float
- Default: 0.9
- The input value that is provided in the Top-P Slider component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Результат" Audio component.

[1]: - Type: str
- The output value that appears in the "Статус" Textbox component.



### API Name: /stop_generation_fn
Description: Остановка генерации.

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	api_name="/stop_generation_fn"
)
print(result)
```

Accepts 0 parameters:



Returns 1 element:

- Type: str
- The output value that appears in the "Статус" Textbox component.



### API Name: /load_voice_preset


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	voice_name="RU_Famale_ALEKSANDRA_KARELьSKIKh",
	api_name="/load_voice_preset"
)
print(result)
```

Accepts 1 parameter:

voice_name:
- Type: Literal['-- Загрузить свой --', 'Arabic_Female', 'EN_Female_Anime_Girl', 'EN_Male_Mortal_Kombat', 'English_Female', 'English_Female2', 'French_Female', 'German_Male', 'Japanese_Male', 'Korean_Female', 'RU_Famale_ALEKSANDRA_KARELьSKIKh', 'RU_Famale_ALEKSANDRA_KOSTINSKAYa', 'RU_Famale_ALEKSANDRA_KOZhEVNIKOVA', 'RU_Famale_ALEKSANDRA_KURAGINA', 'RU_Famale_ALEKSANDRA_NAZAROVA', 'RU_Famale_ALEKSANDRA_OSTROUKhOVA', 'RU_Famale_ALEKSANDRA_VERKhOShANSKAYa', 'RU_Female_AliExpress', 'RU_Female_Cheburashka', 'RU_Female_IngridOlerinskaya_Judy', 'RU_Female_Kropina_YouTube', 'RU_Female_YandexAlisa', 'RU_MALE_Zadornov(koncert)', 'RU_Male_AbdulovV', 'RU_Male_Buhmin_AudioBook', 'RU_Male_CheburashkaMovie_2023', 'RU_Male_Craster_YouTube', 'RU_Male_Deadpool', 'RU_Male_Denis_Kolesnikov', 'RU_Male_Druzhko_Sergey', 'RU_Male_Gabidullin_ruslan', 'RU_Male_Goblin_Puchkov', 'RU_Male_Gorshok', 'RU_Male_JohnnySilverhand', 'RU_Male_Livanov_SherlockHolmes', 'RU_Male_Minaev', 'RU_Male_Nagiev', 'RU_Male_OptimusPrime', 'RU_Male_Sergey_Chihachev_Terminator', 'RU_Male_SpoungeBob', 'RU_Male_Tinkoff', 'RU_Male_Volodarskiy', 'RU_Male_Vsevolod-Kuznecov', 'RU_Male_Yuri_The_Professional', 'RU_Male_Zajcev-Vladimir', 'RU_Male_Zhirinovsky', 'Ru_Male_KlimZhukov', 'Ru_Male_SSSR_Matroskin', 'Ru_Male_SSSR_Vinipuh', 'Ru_Male_USSR_DOCS', 'Spanish_Male']
- Default: "RU_Famale_ALEKSANDRA_KARELьSKIKh"
- The input value that is provided in the Выбрать голос из библиотеки Dropdown component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Референсное аудио (голос для клонирования)" Audio component.

[1]: - Type: str
- The output value that appears in the "Текст референсного аудио" Textbox component.



### API Name: /refresh_voice_list


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	api_name="/refresh_voice_list"
)
print(result)
```

Accepts 0 parameters:



Returns 1 element:

- Type: Literal['-- Загрузить свой --', 'Arabic_Female', 'EN_Female_Anime_Girl', 'EN_Male_Mortal_Kombat', 'English_Female', 'English_Female2', 'French_Female', 'German_Male', 'Japanese_Male', 'Korean_Female', 'RU_Famale_ALEKSANDRA_KARELьSKIKh', 'RU_Famale_ALEKSANDRA_KOSTINSKAYa', 'RU_Famale_ALEKSANDRA_KOZhEVNIKOVA', 'RU_Famale_ALEKSANDRA_KURAGINA', 'RU_Famale_ALEKSANDRA_NAZAROVA', 'RU_Famale_ALEKSANDRA_OSTROUKhOVA', 'RU_Famale_ALEKSANDRA_VERKhOShANSKAYa', 'RU_Female_AliExpress', 'RU_Female_Cheburashka', 'RU_Female_IngridOlerinskaya_Judy', 'RU_Female_Kropina_YouTube', 'RU_Female_YandexAlisa', 'RU_MALE_Zadornov(koncert)', 'RU_Male_AbdulovV', 'RU_Male_Buhmin_AudioBook', 'RU_Male_CheburashkaMovie_2023', 'RU_Male_Craster_YouTube', 'RU_Male_Deadpool', 'RU_Male_Denis_Kolesnikov', 'RU_Male_Druzhko_Sergey', 'RU_Male_Gabidullin_ruslan', 'RU_Male_Goblin_Puchkov', 'RU_Male_Gorshok', 'RU_Male_JohnnySilverhand', 'RU_Male_Livanov_SherlockHolmes', 'RU_Male_Minaev', 'RU_Male_Nagiev', 'RU_Male_OptimusPrime', 'RU_Male_Sergey_Chihachev_Terminator', 'RU_Male_SpoungeBob', 'RU_Male_Tinkoff', 'RU_Male_Volodarskiy', 'RU_Male_Vsevolod-Kuznecov', 'RU_Male_Yuri_The_Professional', 'RU_Male_Zajcev-Vladimir', 'RU_Male_Zhirinovsky', 'Ru_Male_KlimZhukov', 'Ru_Male_SSSR_Matroskin', 'Ru_Male_SSSR_Vinipuh', 'Ru_Male_USSR_DOCS', 'Spanish_Male']
- The output value that appears in the "Выбрать голос из библиотеки" Dropdown component.



### API Name: /load_cloud_list_vc


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	api_name="/load_cloud_list_vc"
)
print(result)
```

Accepts 0 parameters:



Returns tuple of 2 elements:

[0]: - Type: str
- The output value that appears in the "Статус" Textbox component.

[1]: - Type: list[Literal[]]
- The output value that appears in the "Доступные голоса" Checkboxgroup component.



### API Name: /download_selected_voices_vc


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	selected=[],
	api_name="/download_selected_voices_vc"
)
print(result)
```

Accepts 1 parameter:

selected:
- Type: list[Literal[]]
- Default: []
- The input value that is provided in the Доступные голоса Checkboxgroup component. 

Returns 1 element:

- Type: str
- The output value that appears in the "Результат загрузки" Textbox component.



### API Name: /generate_voice_clone
Description: Клонирование голоса (стриминг).

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	ref_audio=handle_file('http://127.0.0.1:7860/gradio_api/file=D:\\Work\\ML\\Voice\\Qwen3-TTS_portable_rus-main\\portable\\temp\\5b13bae6801a5289f96d02f02f1590a7bf68c177488058de94ba6511e74c1b7c\\audio.wav'),
	ref_text="",
	target_text="Ничего себе, мой голос клонированный с помощью Qwen3-TTS звучит почти как настоящий!",
	language="Авто (определить автоматически)",
	use_xvector_only=True,
	model_size="1.7B",
	max_tokens=2048,
	temperature=0.7,
	top_p=0.9,
	api_name="/generate_voice_clone"
)
print(result)
```

Accepts 9 parameters:

ref_audio:
- Type: filepath
- Default: handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')
- The input value that is provided in the Референсное аудио (голос для клонирования) Audio component. The FileData class is a subclass of the GradioModel class that represents a file object within a Gradio interface. It is used to store file data and metadata when a file is uploaded.

Attributes:
    path: The server file path where the file is stored.
    url: The normalized server URL pointing to the file.
    size: The size of the file in bytes.
    orig_name: The original filename before upload.
    mime_type: The MIME type of the file.
    is_stream: Indicates whether the file is a stream.
    meta: Additional metadata used internally (should not be changed).

ref_text:
- Type: str
- Default: "я была крохой когда у нас отняли ферму до сих пор помню как нас вели по грязным улицам к нашему новому дому в гетто моя тётя научила меня как нужно бороться с нацистами жаль во сне не встретитесь вы бы ей понравились сегодня ясный день но до вашего приезда дождь лил несколько недель очень странно мне бы ваши силы девочки думаю у нас есть все шансы прогнать нацистов даже странно что сегодня нет дождя видимо скоро пойдёт"
- The input value that is provided in the Текст референсного аудио Textbox component. 

target_text:
- Type: str
- Default: "Ничего себе, мой голос клонированный с помощью Qwen3-TTS звучит почти как настоящий!"
- The input value that is provided in the Текст для синтеза Textbox component. 

language:
- Type: Literal['Авто (определить автоматически)', 'Русский', 'Английский', 'Китайский', 'Японский', 'Корейский', 'Французский', 'Немецкий', 'Испанский', 'Португальский', 'Итальянский']
- Default: "Авто (определить автоматически)"
- The input value that is provided in the Язык Dropdown component. 

use_xvector_only:
- Type: bool
- Default: False
- The input value that is provided in the Только x-vector (без текста референса, качество ниже) Checkbox component. 

model_size:
- Type: Literal['0.6B', '1.7B']
- Default: "1.7B"
- The input value that is provided in the Размер модели Dropdown component. 

max_tokens:
- Type: float
- Default: 2048
- The input value that is provided in the Макс. токенов Slider component. 

temperature:
- Type: float
- Default: 0.7
- The input value that is provided in the Температура Slider component. 

top_p:
- Type: float
- Default: 0.9
- The input value that is provided in the Top-P Slider component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Результат" Audio component.

[1]: - Type: str
- The output value that appears in the "Статус" Textbox component.



### API Name: /stop_generation_fn_1
Description: Остановка генерации.

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	api_name="/stop_generation_fn_1"
)
print(result)
```

Accepts 0 parameters:



Returns 1 element:

- Type: str
- The output value that appears in the "Статус" Textbox component.



### API Name: /update_from_preset


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	preset_name="RU_Male_Gorshok",
	api_name="/update_from_preset"
)
print(result)
```

Accepts 1 parameter:

preset_name:
- Type: Literal['-- Загрузить свой --', 'Arabic_Female', 'EN_Female_Anime_Girl', 'EN_Male_Mortal_Kombat', 'English_Female', 'English_Female2', 'French_Female', 'German_Male', 'Japanese_Male', 'Korean_Female', 'RU_Famale_ALEKSANDRA_KARELьSKIKh', 'RU_Famale_ALEKSANDRA_KOSTINSKAYa', 'RU_Famale_ALEKSANDRA_KOZhEVNIKOVA', 'RU_Famale_ALEKSANDRA_KURAGINA', 'RU_Famale_ALEKSANDRA_NAZAROVA', 'RU_Famale_ALEKSANDRA_OSTROUKhOVA', 'RU_Famale_ALEKSANDRA_VERKhOShANSKAYa', 'RU_Female_AliExpress', 'RU_Female_Cheburashka', 'RU_Female_IngridOlerinskaya_Judy', 'RU_Female_Kropina_YouTube', 'RU_Female_YandexAlisa', 'RU_MALE_Zadornov(koncert)', 'RU_Male_AbdulovV', 'RU_Male_Buhmin_AudioBook', 'RU_Male_CheburashkaMovie_2023', 'RU_Male_Craster_YouTube', 'RU_Male_Deadpool', 'RU_Male_Denis_Kolesnikov', 'RU_Male_Druzhko_Sergey', 'RU_Male_Gabidullin_ruslan', 'RU_Male_Goblin_Puchkov', 'RU_Male_Gorshok', 'RU_Male_JohnnySilverhand', 'RU_Male_Livanov_SherlockHolmes', 'RU_Male_Minaev', 'RU_Male_Nagiev', 'RU_Male_OptimusPrime', 'RU_Male_Sergey_Chihachev_Terminator', 'RU_Male_SpoungeBob', 'RU_Male_Tinkoff', 'RU_Male_Volodarskiy', 'RU_Male_Vsevolod-Kuznecov', 'RU_Male_Yuri_The_Professional', 'RU_Male_Zajcev-Vladimir', 'RU_Male_Zhirinovsky', 'Ru_Male_KlimZhukov', 'Ru_Male_SSSR_Matroskin', 'Ru_Male_SSSR_Vinipuh', 'Ru_Male_USSR_DOCS', 'Spanish_Male']
- Default: "RU_Male_Gorshok"
- The input value that is provided in the Пресет голоса Dropdown component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Аудио референса" Audio component.

[1]: - Type: str
- The output value that appears in the "Текст референса (опционально)" Textbox component.



### API Name: /update_from_preset_1


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	preset_name="RU_Female_IngridOlerinskaya_Judy",
	api_name="/update_from_preset_1"
)
print(result)
```

Accepts 1 parameter:

preset_name:
- Type: Literal['-- Загрузить свой --', 'Arabic_Female', 'EN_Female_Anime_Girl', 'EN_Male_Mortal_Kombat', 'English_Female', 'English_Female2', 'French_Female', 'German_Male', 'Japanese_Male', 'Korean_Female', 'RU_Famale_ALEKSANDRA_KARELьSKIKh', 'RU_Famale_ALEKSANDRA_KOSTINSKAYa', 'RU_Famale_ALEKSANDRA_KOZhEVNIKOVA', 'RU_Famale_ALEKSANDRA_KURAGINA', 'RU_Famale_ALEKSANDRA_NAZAROVA', 'RU_Famale_ALEKSANDRA_OSTROUKhOVA', 'RU_Famale_ALEKSANDRA_VERKhOShANSKAYa', 'RU_Female_AliExpress', 'RU_Female_Cheburashka', 'RU_Female_IngridOlerinskaya_Judy', 'RU_Female_Kropina_YouTube', 'RU_Female_YandexAlisa', 'RU_MALE_Zadornov(koncert)', 'RU_Male_AbdulovV', 'RU_Male_Buhmin_AudioBook', 'RU_Male_CheburashkaMovie_2023', 'RU_Male_Craster_YouTube', 'RU_Male_Deadpool', 'RU_Male_Denis_Kolesnikov', 'RU_Male_Druzhko_Sergey', 'RU_Male_Gabidullin_ruslan', 'RU_Male_Goblin_Puchkov', 'RU_Male_Gorshok', 'RU_Male_JohnnySilverhand', 'RU_Male_Livanov_SherlockHolmes', 'RU_Male_Minaev', 'RU_Male_Nagiev', 'RU_Male_OptimusPrime', 'RU_Male_Sergey_Chihachev_Terminator', 'RU_Male_SpoungeBob', 'RU_Male_Tinkoff', 'RU_Male_Volodarskiy', 'RU_Male_Vsevolod-Kuznecov', 'RU_Male_Yuri_The_Professional', 'RU_Male_Zajcev-Vladimir', 'RU_Male_Zhirinovsky', 'Ru_Male_KlimZhukov', 'Ru_Male_SSSR_Matroskin', 'Ru_Male_SSSR_Vinipuh', 'Ru_Male_USSR_DOCS', 'Spanish_Male']
- Default: "RU_Female_IngridOlerinskaya_Judy"
- The input value that is provided in the Пресет голоса Dropdown component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Аудио референса" Audio component.

[1]: - Type: str
- The output value that appears in the "Текст референса (опционально)" Textbox component.



### API Name: /update_from_preset_2


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	preset_name="RU_Male_Livanov_SherlockHolmes",
	api_name="/update_from_preset_2"
)
print(result)
```

Accepts 1 parameter:

preset_name:
- Type: Literal['-- Загрузить свой --', 'Arabic_Female', 'EN_Female_Anime_Girl', 'EN_Male_Mortal_Kombat', 'English_Female', 'English_Female2', 'French_Female', 'German_Male', 'Japanese_Male', 'Korean_Female', 'RU_Famale_ALEKSANDRA_KARELьSKIKh', 'RU_Famale_ALEKSANDRA_KOSTINSKAYa', 'RU_Famale_ALEKSANDRA_KOZhEVNIKOVA', 'RU_Famale_ALEKSANDRA_KURAGINA', 'RU_Famale_ALEKSANDRA_NAZAROVA', 'RU_Famale_ALEKSANDRA_OSTROUKhOVA', 'RU_Famale_ALEKSANDRA_VERKhOShANSKAYa', 'RU_Female_AliExpress', 'RU_Female_Cheburashka', 'RU_Female_IngridOlerinskaya_Judy', 'RU_Female_Kropina_YouTube', 'RU_Female_YandexAlisa', 'RU_MALE_Zadornov(koncert)', 'RU_Male_AbdulovV', 'RU_Male_Buhmin_AudioBook', 'RU_Male_CheburashkaMovie_2023', 'RU_Male_Craster_YouTube', 'RU_Male_Deadpool', 'RU_Male_Denis_Kolesnikov', 'RU_Male_Druzhko_Sergey', 'RU_Male_Gabidullin_ruslan', 'RU_Male_Goblin_Puchkov', 'RU_Male_Gorshok', 'RU_Male_JohnnySilverhand', 'RU_Male_Livanov_SherlockHolmes', 'RU_Male_Minaev', 'RU_Male_Nagiev', 'RU_Male_OptimusPrime', 'RU_Male_Sergey_Chihachev_Terminator', 'RU_Male_SpoungeBob', 'RU_Male_Tinkoff', 'RU_Male_Volodarskiy', 'RU_Male_Vsevolod-Kuznecov', 'RU_Male_Yuri_The_Professional', 'RU_Male_Zajcev-Vladimir', 'RU_Male_Zhirinovsky', 'Ru_Male_KlimZhukov', 'Ru_Male_SSSR_Matroskin', 'Ru_Male_SSSR_Vinipuh', 'Ru_Male_USSR_DOCS', 'Spanish_Male']
- Default: "RU_Male_Livanov_SherlockHolmes"
- The input value that is provided in the Пресет голоса Dropdown component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Аудио референса" Audio component.

[1]: - Type: str
- The output value that appears in the "Текст референса (опционально)" Textbox component.



### API Name: /update_from_preset_3


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	preset_name="RU_Male_Yuri_The_Professional",
	api_name="/update_from_preset_3"
)
print(result)
```

Accepts 1 parameter:

preset_name:
- Type: Literal['-- Загрузить свой --', 'Arabic_Female', 'EN_Female_Anime_Girl', 'EN_Male_Mortal_Kombat', 'English_Female', 'English_Female2', 'French_Female', 'German_Male', 'Japanese_Male', 'Korean_Female', 'RU_Famale_ALEKSANDRA_KARELьSKIKh', 'RU_Famale_ALEKSANDRA_KOSTINSKAYa', 'RU_Famale_ALEKSANDRA_KOZhEVNIKOVA', 'RU_Famale_ALEKSANDRA_KURAGINA', 'RU_Famale_ALEKSANDRA_NAZAROVA', 'RU_Famale_ALEKSANDRA_OSTROUKhOVA', 'RU_Famale_ALEKSANDRA_VERKhOShANSKAYa', 'RU_Female_AliExpress', 'RU_Female_Cheburashka', 'RU_Female_IngridOlerinskaya_Judy', 'RU_Female_Kropina_YouTube', 'RU_Female_YandexAlisa', 'RU_MALE_Zadornov(koncert)', 'RU_Male_AbdulovV', 'RU_Male_Buhmin_AudioBook', 'RU_Male_CheburashkaMovie_2023', 'RU_Male_Craster_YouTube', 'RU_Male_Deadpool', 'RU_Male_Denis_Kolesnikov', 'RU_Male_Druzhko_Sergey', 'RU_Male_Gabidullin_ruslan', 'RU_Male_Goblin_Puchkov', 'RU_Male_Gorshok', 'RU_Male_JohnnySilverhand', 'RU_Male_Livanov_SherlockHolmes', 'RU_Male_Minaev', 'RU_Male_Nagiev', 'RU_Male_OptimusPrime', 'RU_Male_Sergey_Chihachev_Terminator', 'RU_Male_SpoungeBob', 'RU_Male_Tinkoff', 'RU_Male_Volodarskiy', 'RU_Male_Vsevolod-Kuznecov', 'RU_Male_Yuri_The_Professional', 'RU_Male_Zajcev-Vladimir', 'RU_Male_Zhirinovsky', 'Ru_Male_KlimZhukov', 'Ru_Male_SSSR_Matroskin', 'Ru_Male_SSSR_Vinipuh', 'Ru_Male_USSR_DOCS', 'Spanish_Male']
- Default: "RU_Male_Yuri_The_Professional"
- The input value that is provided in the Пресет голоса Dropdown component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Аудио референса" Audio component.

[1]: - Type: str
- The output value that appears in the "Текст референса (опционально)" Textbox component.



### API Name: /update_speaker_visibility


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	num=2,
	api_name="/update_speaker_visibility"
)
print(result)
```

Accepts 1 parameter:

num:
- Type: float
- Default: 2
- The input value that is provided in the Количество дикторов Slider component. 

Returns 1 element:





### API Name: /multi_speaker_wrapper


```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	script="Speaker 0: Привет! Ты уже подписался на канал нейро-софт?
Speaker 1: Да, там регулярно выходят портативные версии полезных нейросетей!
Speaker 0: Это точно, а еще классные мемы!",
	num_speakers=2,
	audio0=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
	audio1=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
	audio2=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
	audio3=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
	text0="Я жив, покуда я верю в чудо
Но должен буду я умереть
Мне очень грустно, что в сердце пусто
Все мои чувства забрал медведь
Моя судьба мне неподвластна
Любовь моя, как смерть, опасна
Погаснет день, луна проснётся
И снова зверь во мне очнётся
Забрали чары души покой
Возник вопрос: "Кто я такой"
Мой бедный разум дошёл не сразу
До странной мысли: я человек
Колдун был пьяный, весьма упрямый
Его не видеть бы, да, мне вовек
Моя судьба мне неподвластна
Любовь моя, как смерть опасна
Я был медведем, проблем не знал
Зачем людских кровей я стал
И оборвётся тут словно нить
Мой дар – на двух, хмм ногах ходить
Хой!",
	text1="﻿Этот хер поставил тебе беушный хром.
Мало того, что риэл скин облез, так вместо него еще и синтетическое говно,
которое на старый кафель похоже.
Ты не идиот!
Как тебе кажется, почему девушки, которые сюда приходят, позволяют тебе их лапать, а?
Потому что ты заботливый и неотразимый?
Хуй там!",
	text2="﻿Земля вращается вокруг солнца, но мне в моем деле это не пригодится.
Мистер Ватсон, я могу вас утешить.
Дело в том, что таких людей, как я, в мире очень немного.
Может быть, даже я такой один.
Скажите, Ватсон, вам никогда никто не встречался
из этих милых джентльменов?",
	text3="﻿Здравствуйте, мои дорогие зрители. Мы смотрим традиционный спорт эскимосов на физическую выносливость и способность выдерживать боль - ушная тяга. Уберите женщин от экранов! Брутальность здесь зашкаливает! Ваша девушка может начать задавать вам вопросы, задумываться, а вы бы так смогли? Это невыгодно! Молодой человек в красном аж отвлекся от смартфона, от мемасиков! Шок! Крутиться нельзя, только тянуть. А у кого первым порвется или соскочит веревочка - тот проиграл. 60-сантиметровая вощеная нить очень похожа на зубную, оборачивается вокруг ушей и доставляет некоторый дискомфорт. Молодой пытался, но опыт не пропьешь. Но видно, что дяденька уже на последних щах. Одно очко он заработал, но волосы уже встали дыбом. Он понимает, что ему пора прощаться с этим спортом. А! Последние щи сливают! Черпает со дна кастрюли и побеждает! Он показал, что эскимосы не сдаются! Господи, дяденьке надо помочь! А вот суперчемпион ушной тяги - Джеймс Ламп, 67-го года рождения, капитан китобойного судна, смотрит на тебя как бумер на зумера. Из города Анкоридж в Аляске. Он чемпион не только по ушной тяге, но и по удержанию ушами огромных весов. Каждый год проводятся Всемирные эскимосские индейские Олимпийские игры. Вот этот дяденька молодец позитивный. И Джеймс Ламп доминирует на этих играх. Успех среди женщин, у него шестеро детей. Своими ушами он просто притягивает к себе людей. Но появился Дрю Дюбэри, по-нашему Андрей. В тяге ушей ему помогает ведьма, которая вселилась в него в детстве. Он пришелся чемпионом. Ой, говорит, а это что? Я и не почувствовал. Эти игры изначально возникли для того, чтобы подготавливать детей к строгим условиям северной местности, тренировать их координацию, силу и сообразительность. Ой нет, перед чемпионом остался только дедушка. Пожалуйста, не мучай его! Последний паладин перед чемпионом пал. Ну хорошо, давайте смотреть финал. От этого матча зависит судьба человечества. Человечество в надежных ушах. Молодой Андрей попадает на третье место. А мы еще продолжим следить за Всемирными эскимоско-индейскими Олимпийскими играми. Дорогие зрители, вам здоровья, добра, любви и процветания! До свидания!",
	language="Авто (определить автоматически)",
	model_size="1.7B",
	max_tokens=2048,
	temperature=0.7,
	top_p=0.9,
	api_name="/multi_speaker_wrapper"
)
print(result)
```

Accepts 15 parameters:

script:
- Type: str
- Default: "Speaker 0: Привет! Ты уже подписался на канал нейро-софт?
Speaker 1: Да, там регулярно выходят портативные версии полезных нейросетей!
Speaker 0: Это точно, а еще классные мемы!"
- The input value that is provided in the Сценарий диалога Textbox component. 

num_speakers:
- Type: float
- Default: 2
- The input value that is provided in the Количество дикторов Slider component. 

audio0:
- Type: filepath
- Default: handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')
- The input value that is provided in the Аудио референса Audio component. The FileData class is a subclass of the GradioModel class that represents a file object within a Gradio interface. It is used to store file data and metadata when a file is uploaded.

Attributes:
    path: The server file path where the file is stored.
    url: The normalized server URL pointing to the file.
    size: The size of the file in bytes.
    orig_name: The original filename before upload.
    mime_type: The MIME type of the file.
    is_stream: Indicates whether the file is a stream.
    meta: Additional metadata used internally (should not be changed).

audio1:
- Type: filepath
- Default: handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')
- The input value that is provided in the Аудио референса Audio component. The FileData class is a subclass of the GradioModel class that represents a file object within a Gradio interface. It is used to store file data and metadata when a file is uploaded.

Attributes:
    path: The server file path where the file is stored.
    url: The normalized server URL pointing to the file.
    size: The size of the file in bytes.
    orig_name: The original filename before upload.
    mime_type: The MIME type of the file.
    is_stream: Indicates whether the file is a stream.
    meta: Additional metadata used internally (should not be changed).

audio2:
- Type: filepath
- Default: handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')
- The input value that is provided in the Аудио референса Audio component. The FileData class is a subclass of the GradioModel class that represents a file object within a Gradio interface. It is used to store file data and metadata when a file is uploaded.

Attributes:
    path: The server file path where the file is stored.
    url: The normalized server URL pointing to the file.
    size: The size of the file in bytes.
    orig_name: The original filename before upload.
    mime_type: The MIME type of the file.
    is_stream: Indicates whether the file is a stream.
    meta: Additional metadata used internally (should not be changed).

audio3:
- Type: filepath
- Default: handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')
- The input value that is provided in the Аудио референса Audio component. The FileData class is a subclass of the GradioModel class that represents a file object within a Gradio interface. It is used to store file data and metadata when a file is uploaded.

Attributes:
    path: The server file path where the file is stored.
    url: The normalized server URL pointing to the file.
    size: The size of the file in bytes.
    orig_name: The original filename before upload.
    mime_type: The MIME type of the file.
    is_stream: Indicates whether the file is a stream.
    meta: Additional metadata used internally (should not be changed).

text0:
- Type: str
- Default: "Я жив, покуда я верю в чудо
Но должен буду я умереть
Мне очень грустно, что в сердце пусто
Все мои чувства забрал медведь
Моя судьба мне неподвластна
Любовь моя, как смерть, опасна
Погаснет день, луна проснётся
И снова зверь во мне очнётся
Забрали чары души покой
Возник вопрос: "Кто я такой"
Мой бедный разум дошёл не сразу
До странной мысли: я человек
Колдун был пьяный, весьма упрямый
Его не видеть бы, да, мне вовек
Моя судьба мне неподвластна
Любовь моя, как смерть опасна
Я был медведем, проблем не знал
Зачем людских кровей я стал
И оборвётся тут словно нить
Мой дар – на двух, хмм ногах ходить
Хой!"
- The input value that is provided in the Текст референса (опционально) Textbox component. 

text1:
- Type: str
- Default: "﻿Этот хер поставил тебе беушный хром.
Мало того, что риэл скин облез, так вместо него еще и синтетическое говно,
которое на старый кафель похоже.
Ты не идиот!
Как тебе кажется, почему девушки, которые сюда приходят, позволяют тебе их лапать, а?
Потому что ты заботливый и неотразимый?
Хуй там!"
- The input value that is provided in the Текст референса (опционально) Textbox component. 

text2:
- Type: str
- Default: "﻿Земля вращается вокруг солнца, но мне в моем деле это не пригодится.
Мистер Ватсон, я могу вас утешить.
Дело в том, что таких людей, как я, в мире очень немного.
Может быть, даже я такой один.
Скажите, Ватсон, вам никогда никто не встречался
из этих милых джентльменов?"
- The input value that is provided in the Текст референса (опционально) Textbox component. 

text3:
- Type: str
- Default: "﻿Здравствуйте, мои дорогие зрители. Мы смотрим традиционный спорт эскимосов на физическую выносливость и способность выдерживать боль - ушная тяга. Уберите женщин от экранов! Брутальность здесь зашкаливает! Ваша девушка может начать задавать вам вопросы, задумываться, а вы бы так смогли? Это невыгодно! Молодой человек в красном аж отвлекся от смартфона, от мемасиков! Шок! Крутиться нельзя, только тянуть. А у кого первым порвется или соскочит веревочка - тот проиграл. 60-сантиметровая вощеная нить очень похожа на зубную, оборачивается вокруг ушей и доставляет некоторый дискомфорт. Молодой пытался, но опыт не пропьешь. Но видно, что дяденька уже на последних щах. Одно очко он заработал, но волосы уже встали дыбом. Он понимает, что ему пора прощаться с этим спортом. А! Последние щи сливают! Черпает со дна кастрюли и побеждает! Он показал, что эскимосы не сдаются! Господи, дяденьке надо помочь! А вот суперчемпион ушной тяги - Джеймс Ламп, 67-го года рождения, капитан китобойного судна, смотрит на тебя как бумер на зумера. Из города Анкоридж в Аляске. Он чемпион не только по ушной тяге, но и по удержанию ушами огромных весов. Каждый год проводятся Всемирные эскимосские индейские Олимпийские игры. Вот этот дяденька молодец позитивный. И Джеймс Ламп доминирует на этих играх. Успех среди женщин, у него шестеро детей. Своими ушами он просто притягивает к себе людей. Но появился Дрю Дюбэри, по-нашему Андрей. В тяге ушей ему помогает ведьма, которая вселилась в него в детстве. Он пришелся чемпионом. Ой, говорит, а это что? Я и не почувствовал. Эти игры изначально возникли для того, чтобы подготавливать детей к строгим условиям северной местности, тренировать их координацию, силу и сообразительность. Ой нет, перед чемпионом остался только дедушка. Пожалуйста, не мучай его! Последний паладин перед чемпионом пал. Ну хорошо, давайте смотреть финал. От этого матча зависит судьба человечества. Человечество в надежных ушах. Молодой Андрей попадает на третье место. А мы еще продолжим следить за Всемирными эскимоско-индейскими Олимпийскими играми. Дорогие зрители, вам здоровья, добра, любви и процветания! До свидания!"
- The input value that is provided in the Текст референса (опционально) Textbox component. 

language:
- Type: Literal['Авто (определить автоматически)', 'Русский', 'Английский', 'Китайский', 'Японский', 'Корейский', 'Французский', 'Немецкий', 'Испанский', 'Португальский', 'Итальянский']
- Default: "Авто (определить автоматически)"
- The input value that is provided in the Язык Dropdown component. 

model_size:
- Type: Literal['0.6B', '1.7B']
- Default: "1.7B"
- The input value that is provided in the Размер модели Dropdown component. 

max_tokens:
- Type: float
- Default: 2048
- The input value that is provided in the Макс. токенов Slider component. 

temperature:
- Type: float
- Default: 0.7
- The input value that is provided in the Температура Slider component. 

top_p:
- Type: float
- Default: 0.9
- The input value that is provided in the Top-P Slider component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Результат" Audio component.

[1]: - Type: str
- The output value that appears in the "Статус" Textbox component.



### API Name: /stop_generation_fn_2
Description: Остановка генерации.

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	api_name="/stop_generation_fn_2"
)
print(result)
```

Accepts 0 parameters:



Returns 1 element:

- Type: str
- The output value that appears in the "Статус" Textbox component.



### API Name: /generate_voice_design
Description: Генерация речи с дизайном голоса (стриминг).

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	text="Привет! Как твои дела? Это демонстрация синтеза речи Qwen3-TTS от канала Нейро-софт.",
	language="Русский",
	voice_description="Young female voice, warm and friendly, speaking with enthusiasm",
	model_size="1.7B",
	max_tokens=2048,
	temperature=0.7,
	top_p=0.9,
	api_name="/generate_voice_design"
)
print(result)
```

Accepts 7 parameters:

text:
- Type: str
- Default: "Привет! Как твои дела? Это демонстрация синтеза речи Qwen3-TTS от канала Нейро-софт."
- The input value that is provided in the Текст для синтеза Textbox component. 

language:
- Type: Literal['Авто (определить автоматически)', 'Русский', 'Английский', 'Китайский', 'Японский', 'Корейский', 'Французский', 'Немецкий', 'Испанский', 'Португальский', 'Итальянский']
- Default: "Русский"
- The input value that is provided in the Язык Dropdown component. 

voice_description:
- Type: str
- Default: "Young female voice, warm and friendly, speaking with enthusiasm"
- The input value that is provided in the Описание голоса (лучше на английском) Textbox component. 

model_size:
- Type: Literal['1.7B']
- Default: "1.7B"
- The input value that is provided in the Размер модели Dropdown component. 

max_tokens:
- Type: float
- Default: 2048
- The input value that is provided in the Макс. токенов Slider component. 

temperature:
- Type: float
- Default: 0.7
- The input value that is provided in the Температура Slider component. 

top_p:
- Type: float
- Default: 0.9
- The input value that is provided in the Top-P Slider component. 

Returns tuple of 2 elements:

[0]: - Type: filepath
- The output value that appears in the "Результат" Audio component.

[1]: - Type: str
- The output value that appears in the "Статус" Textbox component.



### API Name: /stop_generation_fn_3
Description: Остановка генерации.

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	api_name="/stop_generation_fn_3"
)
print(result)
```

Accepts 0 parameters:



Returns 1 element:

- Type: str
- The output value that appears in the "Статус" Textbox component.

