"""Real integration tests for summarization inference and API.

These tests:
- load actual HuggingFace models
- run real summarization inference
- verify language switching
- verify FastAPI `/summarize` endpoint
- guard against transformers pipeline regressions
"""

import unittest

from fastapi.testclient import TestClient

from api.dependencies import get_summarization_model
from api.server import app
from models.summarization.interface import SummarizationModel


LONG_ENGLISH_TEXT = """
Open-source AI systems can help teams summarize long product updates,
meeting notes, and release plans. When the input is noisy or repetitive,
a stable summarizer should still return a short, readable result.

This regression test keeps the example long enough to resemble a real
request body for the API endpoint. The goal is to verify that model
loading, language detection, generation, and endpoint wiring all stay healthy.
"""


LONG_VIETNAMESE_TEXT = """
Hệ thống AI mã nguồn mở có thể giúp đội ngũ tóm tắt các bản cập nhật sản phẩm dài,
ghi chú họp và kế hoạch phát hành. Khi đầu vào nhiều nhiễu hoặc lặp lại,
bộ tóm tắt vẫn cần trả về một kết quả ngắn gọn và dễ đọc.

Mẫu dữ liệu kiểm thử này được viết dài hơn để giống một request thực tế gửi vào API.
Mục tiêu là kiểm tra việc tải model, nhận diện ngôn ngữ,
sinh văn bản và endpoint vẫn hoạt động bình thường.
"""


class SummarizationRealIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[INIT] Loading real summarization model...")
        cls.model = SummarizationModel()

    # --------------------------------------------------
    # Direct inference tests
    # --------------------------------------------------

    def test_real_english_inference(self):
        result = self.model.infer(LONG_ENGLISH_TEXT)

        print("\n[REAL ENGLISH SUMMARY]")
        print(result)

        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)

        summary = result["summary"]

        self.assertTrue(summary.strip())
        self.assertGreater(len(summary), 10)

        self.assertEqual(self.model.current_language, "en")

    def test_real_vietnamese_inference(self):
        result = self.model.infer(LONG_VIETNAMESE_TEXT)

        print("\n[REAL VIETNAMESE SUMMARY]")
        print(result)

        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)

        summary = result["summary"]

        self.assertTrue(summary.strip())
        self.assertGreater(len(summary), 10)

        self.assertEqual(self.model.current_language, "vi")

    # --------------------------------------------------
    # API tests
    # --------------------------------------------------

    def test_real_api_summarize_english(self):
        app.dependency_overrides[get_summarization_model] = lambda: self.model
        self.addCleanup(app.dependency_overrides.clear)

        client = TestClient(app)

        response = client.post(
            "/summarize",
            json={"text": LONG_ENGLISH_TEXT}
        )

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        print("\n[REAL API ENGLISH RESPONSE]")
        print(payload)

        self.assertEqual(payload["status"], "success")

        summary = payload["data"]["summary"]

        self.assertTrue(summary.strip())
        self.assertGreater(len(summary), 10)

    def test_real_api_summarize_vietnamese(self):
        app.dependency_overrides[get_summarization_model] = lambda: self.model
        self.addCleanup(app.dependency_overrides.clear)

        client = TestClient(app)

        response = client.post(
            "/summarize",
            json={"text": LONG_VIETNAMESE_TEXT}
        )

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        print("\n[REAL API VIETNAMESE RESPONSE]")
        print(payload)

        self.assertEqual(payload["status"], "success")

        summary = payload["data"]["summary"]

        self.assertTrue(summary.strip())
        self.assertGreater(len(summary), 10)

    # --------------------------------------------------
    # Regression check
    # --------------------------------------------------

    def test_pipeline_task_is_generic(self):
        self.assertEqual(
            self.model.current_pipeline_task,
            "text2text-generation"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)