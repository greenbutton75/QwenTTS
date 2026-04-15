import ast
from pathlib import Path
import unittest


class ServerSpliceAttemptLimitTests(unittest.TestCase):
    def test_splice_path_uses_splice_attempt_limit_constant(self) -> None:
        tree = ast.parse(Path("server/app.py").read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "generate_voice_with_similarity_retry":
                continue
            keyword_names = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            if "max_attempts" not in keyword_names:
                continue
            max_attempts = keyword_names["max_attempts"]
            self.assertIsInstance(max_attempts, ast.Name)
            self.assertEqual(max_attempts.id, "GREETING_SPLICE_MAX_ATTEMPTS")
            return

        self.fail("generate_voice_with_similarity_retry(max_attempts=GREETING_SPLICE_MAX_ATTEMPTS) not found in server/app.py")

    def test_splice_path_passes_custom_generate_configs(self) -> None:
        tree = ast.parse(Path("server/app.py").read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "generate_voice_with_similarity_retry":
                continue
            keyword_names = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            initial_generate_config = keyword_names.get("initial_generate_config")
            retry_generate_config = keyword_names.get("retry_generate_config")
            self.assertIsInstance(initial_generate_config, ast.Name)
            self.assertEqual(initial_generate_config.id, "splice_initial_generate_config")
            self.assertIsInstance(retry_generate_config, ast.Name)
            self.assertEqual(retry_generate_config.id, "splice_retry_generate_config")
            return

        self.fail("generate_voice_with_similarity_retry(custom splice generate configs) not found in server/app.py")


if __name__ == "__main__":
    unittest.main()
