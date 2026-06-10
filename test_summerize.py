"""Real integration tests for summarization inference and API."""

import io
import logging
import sys
import unittest
import warnings

from fastapi.testclient import TestClient

from api.dependencies import get_summarization_model
from api.server import app
from models.summarization.interface import SummarizationModel
import threading

# =========================================================
# SUPPRESS ALL NOISE BEFORE ANYTHING ELSE
# =========================================================

# Silence all loggers
logging.disable(logging.WARNING)

# Silence all Python warnings (including C-extension DeprecationWarnings)
warnings.filterwarnings("ignore")


# Custom stdout: buffer by full line, only pass through lines with our prefixes
class _SilentStdout(io.StringIO):
    _ALLOW = ("[INIT]", "[EN]", "[VI]", "[API", "FAILED", "ERROR")
    _buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if any(tag in line for tag in self._ALLOW):
                sys.__stdout__.write(line + "\n")

    def flush(self):
        # Flush remaining buffer without newline (e.g. final line)
        if self._buf and any(tag in self._buf for tag in self._ALLOW):
            sys.__stdout__.write(self._buf)
            self._buf = ""
        sys.__stdout__.flush()

sys.stdout = _SilentStdout()


# =========================================================
# FIXTURES
# =========================================================

LONG_ENGLISH_TEXT = """
Open-source AI systems can help teams summarize long product updates,
meeting notes, and release plans. When the input is noisy or repetitive,
a stable summarizer should still return a short, readable result.

This regression test keeps the example long enough to resemble a real
request body for the API endpoint. The goal is to verify that model
loading, language detection, generation, and endpoint wiring all stay healthy.
"""

LONG_VIETNAMESE_TEXT = """
Theo báo cáo của Bộ Giáo dục và Đào tạo, năm học 2024-2025 có hơn 23 triệu học sinh
trên cả nước tham gia học tập tại các trường phổ thông công lập và tư thục.
Bộ đã triển khai chương trình giáo dục phổ thông mới nhằm nâng cao chất lượng dạy và học,
đồng thời tăng cường ứng dụng công nghệ thông tin trong quản lý và giảng dạy.

Các địa phương cũng tích cực đầu tư cơ sở vật chất, trang thiết bị dạy học hiện đại
để đáp ứng yêu cầu đổi mới giáo dục toàn diện trong giai đoạn mới.
Chính phủ đặt mục tiêu đến năm 2030, tỷ lệ học sinh hoàn thành chương trình trung học
đạt trên 90% ở tất cả các tỉnh thành trên toàn quốc.
"""

DEGENERATE_TEXT = "ILLAR ILLAR ILLAR ILLAR ILLAR ILLAR ILLAR ILLAR ILLAR ILLAR"

UNKNOWN_LANG_TEXT = "kono bun wa nihongo desu. tesuto desu."


# =========================================================
# HELPERS
# =========================================================

def _assert_valid_summary(test: unittest.TestCase, result) -> str:
    test.assertIsInstance(result, dict, "infer() must return a dict")
    test.assertIn("summary", result, "result must contain 'summary' key")
    summary = result["summary"]
    test.assertIsInstance(summary, str, "summary must be a string")
    test.assertTrue(summary.strip(), "summary must not be empty")
    test.assertGreater(len(summary), 10, "summary must be > 10 chars")
    return summary


# =========================================================
# DIRECT MODEL TESTS
# =========================================================

class TestSummarizationInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.__stdout__.write("\n[INIT] Loading model for inference tests...\n")
        cls.model = SummarizationModel()
        sys.__stdout__.write("[INIT] Ready.\n\n")

    # English

    def test_english_inference_returns_valid_summary(self):
        result = self.model.infer(LONG_ENGLISH_TEXT)
        summary = _assert_valid_summary(self, result)
        sys.__stdout__.write(f"  [EN] {summary}\n")

    def test_english_inference_sets_current_language(self):
        self.model.infer(LONG_ENGLISH_TEXT)
        self.assertEqual(self.model.current_language, "en")

    def test_english_inference_uses_bart_model(self):
        self.model.infer(LONG_ENGLISH_TEXT)
        self.assertIn("bart", self.model.model_path.lower())

    # Vietnamese tests

    def test_vietnamese_inference_returns_valid_summary(self):
        result = self.model.infer(LONG_VIETNAMESE_TEXT)
        summary = _assert_valid_summary(self, result)
        sys.__stdout__.write(f"  [VI] {summary}\n")

    def test_vietnamese_inference_sets_current_language(self):
        self.model.infer(LONG_VIETNAMESE_TEXT)
        self.assertEqual(self.model.current_language, "vi")

    def test_vietnamese_inference_uses_vit5_model(self):
        self.model.infer(LONG_VIETNAMESE_TEXT)
        self.assertIn("vit5", self.model.model_path.lower())

    def test_vietnamese_output_is_not_degenerate(self):
        result = self.model.infer(LONG_VIETNAMESE_TEXT)
        summary = _assert_valid_summary(self, result)
        self.assertFalse(
            self.model._looks_degenerate(summary),
            f"VI summary is degenerate: {summary[:120]}"
        )

    # Language switching

    def test_language_switch_en_to_vi(self):
        self.model.infer(LONG_ENGLISH_TEXT)
        self.assertEqual(self.model.current_language, "en")
        self.model.infer(LONG_VIETNAMESE_TEXT)
        self.assertEqual(self.model.current_language, "vi")

    def test_language_switch_vi_to_en(self):
        self.model.infer(LONG_VIETNAMESE_TEXT)
        result = self.model.infer(LONG_ENGLISH_TEXT)
        self.assertEqual(self.model.current_language, "en")
        _assert_valid_summary(self, result)

    def test_repeated_same_language_does_not_reload_model(self):
        self.model.infer(LONG_ENGLISH_TEXT)
        path_before = self.model.model_path
        self.model.infer(LONG_ENGLISH_TEXT)
        self.assertEqual(path_before, self.model.model_path)

    # Edge cases

    def test_empty_string_returns_false(self):
        self.assertFalse(self.model.infer(""))

    def test_whitespace_only_returns_false(self):
        self.assertFalse(self.model.infer("   \n\t  "))

    def test_unknown_language_is_handled_as_english(self):
        result = self.model.infer(UNKNOWN_LANG_TEXT)
        self.assertTrue(result is False or isinstance(result, dict))

    def test_degenerate_en_input_does_not_crash(self):
        self.model.infer(LONG_ENGLISH_TEXT)
        self.assertIsInstance(self.model.summarize(DEGENERATE_TEXT), str)

    def test_degenerate_vi_input_does_not_crash(self):
        self.model.infer(LONG_VIETNAMESE_TEXT)
        self.assertIsInstance(self.model.summarize(DEGENERATE_TEXT), str)

    def test_bytes_input_is_handled(self):
        decoded = self.model.process_input(LONG_ENGLISH_TEXT.encode("utf-8"))
        self.assertIsInstance(decoded, str)
        self.assertTrue(decoded.strip())

    def test_instruction_wrapper_is_stripped(self):
        wrapped = (
            "Summarize the following text into at most 5 key points, "
            "concise and clear, preserving important information. "
            + LONG_ENGLISH_TEXT
        )
        _assert_valid_summary(self, self.model.infer(wrapped))

    # Pipeline state

    def test_bart_pipeline_is_none(self):
        self.model.infer(LONG_ENGLISH_TEXT)
        self.assertIsNone(self.model.summarizer)
        self.assertIsNone(self.model.current_pipeline_task)

    def test_vit5_pipeline_is_none_or_callable(self):
        _assert_valid_summary(self, self.model.infer(LONG_VIETNAMESE_TEXT))
        if self.model.summarizer is not None:
            self.assertTrue(callable(self.model.summarizer))

    def test_current_pipeline_task_matches_model_family(self):
        self.model.infer(LONG_ENGLISH_TEXT)
        self.assertIsNone(self.model.current_pipeline_task)
        self.model.infer(LONG_VIETNAMESE_TEXT)
        self.assertIn(self.model.current_pipeline_task, {None, "text2text-generation"})


# =========================================================
# API ENDPOINT TESTS
# =========================================================

class TestSummarizationAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.__stdout__.write("\n[INIT] Loading model for API tests...\n")
        cls.model = SummarizationModel()
        sys.__stdout__.write("[INIT] Ready.\n\n")

    def setUp(self):
        app.dependency_overrides[get_summarization_model] = lambda: self.model
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        app.dependency_overrides.clear()

    # Happy path

    def test_api_english_returns_200(self):
        response = self.client.post("/summarize", json={"text": LONG_ENGLISH_TEXT})
        self.assertEqual(response.status_code, 200)

    def test_api_english_response_shape(self):
        response = self.client.post("/summarize", json={"text": LONG_ENGLISH_TEXT})
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("summary", payload["data"])
        self.assertGreater(len(payload["data"]["summary"]), 10)
        sys.__stdout__.write(f"  [API EN] {payload['data']['summary']}\n")

    def test_api_vietnamese_returns_200(self):
        response = self.client.post("/summarize", json={"text": LONG_VIETNAMESE_TEXT})
        self.assertEqual(response.status_code, 200)

    def test_api_vietnamese_response_shape(self):
        response = self.client.post("/summarize", json={"text": LONG_VIETNAMESE_TEXT})
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("summary", payload["data"])
        self.assertGreater(len(payload["data"]["summary"]), 10)
        sys.__stdout__.write(f"  [API VI] {payload['data']['summary']}\n")

    # Error cases

    def test_api_empty_text_does_not_return_500(self):
        response = self.client.post("/summarize", json={"text": ""})
        self.assertNotEqual(response.status_code, 500)

    def test_api_missing_text_field_returns_422(self):
        response = self.client.post("/summarize", json={})
        self.assertEqual(response.status_code, 422)

    def test_api_unknown_language_does_not_return_500(self):
        response = self.client.post("/summarize", json={"text": UNKNOWN_LANG_TEXT})
        self.assertNotEqual(response.status_code, 500)

    def test_api_very_short_text_does_not_crash(self):
        response = self.client.post("/summarize", json={"text": "hello"})
        self.assertNotEqual(response.status_code, 500)


# =========================================================
# ENTRY POINT
# =========================================================
class TestSummarizationConcurrent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.__stdout__.write("\n[INIT] Loading model for concurrent tests...\n")
        cls.model = SummarizationModel()
        sys.__stdout__.write("[INIT] Ready.\n\n")

    def test_concurrent_en_and_vi_requests(self):
        """2 requests đồng thời — 1 EN, 1 VI — không crash, đều trả kết quả."""
        results = {}
        errors = {}

        def run(key, text):
            try:
                results[key] = self.model.infer(text)
            except Exception as e:
                errors[key] = str(e)

        t1 = threading.Thread(target=run, args=("en", LONG_ENGLISH_TEXT))
        t2 = threading.Thread(target=run, args=("vi", LONG_VIETNAMESE_TEXT))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertFalse(errors, f"Errors during concurrent requests: {errors}")
        self.assertIn("en", results)
        self.assertIn("vi", results)
        _assert_valid_summary(self, results["en"])
        _assert_valid_summary(self, results["vi"])

        sys.__stdout__.write(f"  [EN] {results['en']['summary']}\n")
        sys.__stdout__.write(f"  [VI] {results['vi']['summary']}\n")

    def test_concurrent_same_language(self):
        """3 EN requests đồng thời — không deadlock."""
        results = {}
        errors = {}

        def run(key):
            try:
                results[key] = self.model.infer(LONG_ENGLISH_TEXT)
            except Exception as e:
                errors[key] = str(e)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertFalse(errors, f"Errors: {errors}")
        self.assertEqual(len(results), 3)
        for r in results.values():
            _assert_valid_summary(self, r)
            
if __name__ == "__main__":
    unittest.main(verbosity=1)