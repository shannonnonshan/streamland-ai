"""
API Test Client
Purpose: Test StreamLand AI API endpoints
Usage: python api_test_client.py
"""

import requests
import json
import sys
from pathlib import Path
from typing import Optional

BASE_URL = "http://localhost:8000"


class APIClient:
    """Client for testing StreamLand AI API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def check_health(self) -> bool:
        """Check if API is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> dict:
        """List available models."""
        response = requests.get(f"{self.base_url}/models")
        return response.json()
    
    def summarize(self, text: str, use_rag: bool = False, context_type: str = "general") -> dict:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            use_rag: Whether to use RAG context
            context_type: Type of RAG context
            
        Returns:
            Response JSON
        """
        payload = {
            "text": text,
            "max_length": 150,
            "min_length": 30,
            "use_rag": use_rag,
            "rag_context_type": context_type
        }
        response = requests.post(
            f"{self.base_url}/summarize",
            json=payload
        )
        return response.json()
    
    def transcribe(self, audio_file: str) -> dict:
        """
        Transcribe audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Response JSON
        """
        if not Path(audio_file).exists():
            return {"error": f"File not found: {audio_file}"}
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/transcribe",
                files=files
            )
        return response.json()
    
    def transcribe_and_summarize(
        self,
        audio_file: str,
        use_rag: bool = False,
        context_type: str = "general",
        max_summary_length: int = 150
    ) -> dict:
        """
        Transcribe audio and summarize in one request.
        
        Args:
            audio_file: Path to audio file
            use_rag: Whether to use RAG context
            context_type: Type of RAG context
            max_summary_length: Max length of summary
            
        Returns:
            Response JSON
        """
        if not Path(audio_file).exists():
            return {"error": f"File not found: {audio_file}"}
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {
                'max_summary_length': max_summary_length,
                'min_summary_length': 30,
                'use_rag': str(use_rag).lower(),
                'rag_context_type': context_type
            }
            response = requests.post(
                f"{self.base_url}/pipeline/transcribe-summarize",
                files=files,
                data=data
            )
        return response.json()


def print_response(title: str, response: dict):
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(json.dumps(response, indent=2, ensure_ascii=False))


def test_basic():
    """Test basic API functionality."""
    client = APIClient()
    
    # Check health
    print("\n[TEST 1] Health Check...")
    if client.check_health():
        print("✓ API is running")
    else:
        print("✗ API is not running")
        print("Please start the API server: python api/server.py")
        return
    
    # List models
    print("\n[TEST 2] List Available Models...")
    models = client.list_models()
    print_response("Available Models", models)


def test_summarization():
    """Test summarization endpoint."""
    client = APIClient()
    
    if not client.check_health():
        print("✗ API is not running")
        return
    
    # Test text
    test_text = """
    Artificial intelligence is transforming the world in unprecedented ways.
    From healthcare to transportation, education to entertainment, AI is revolutionizing
    how we live and work. Machine learning algorithms can now recognize images, understand
    natural language, and make complex decisions. Deep learning models have achieved
    superhuman performance in many domains. However, with great power comes great responsibility.
    We must ensure AI systems are developed ethically, transparently, and with proper oversight.
    The future of AI depends on how we address challenges like bias, privacy, and alignment.
    """
    
    # Test without RAG
    print("\n[TEST 3] Summarization without RAG...")
    result = client.summarize(test_text, use_rag=False)
    print_response("Summary (No RAG)", result)
    
    # Test with RAG
    print("\n[TEST 4] Summarization with RAG (StreamLand AI context)...")
    result = client.summarize(test_text, use_rag=True, context_type="streamland_ai")
    print_response("Summary (With RAG)", result)
    
    # Test with technical context
    print("\n[TEST 5] Summarization with RAG (Technical context)...")
    result = client.summarize(test_text, use_rag=True, context_type="technical")
    print_response("Summary (Technical RAG)", result)


def test_pipeline():
    """Test pipeline (transcribe + summarize)."""
    client = APIClient()
    
    if not client.check_health():
        print("✗ API is not running")
        return
    
    # Check for test audio files
    audio_files = [
        "utils/data/audio/testaudio.mp3",
        "utils/data/audio/testaudio-vn.mp3"
    ]
    
    for audio_file in audio_files:
        if Path(audio_file).exists():
            print(f"\n[TEST 6] Pipeline Test ({audio_file})...")
            result = client.transcribe_and_summarize(audio_file, use_rag=True, context_type="streamland_ai")
            if "error" not in result:
                print_response(f"Pipeline Result: {audio_file}", result)
            else:
                print(f"Error: {result['error']}")
            break
    else:
        print("\n[SKIP] No test audio files found")
        print("Place audio files in utils/data/audio/ directory")


def main():
    """Main test runner."""
    print("="*60)
    print("StreamLand AI API Test Client")
    print("="*60)
    
    # Run tests
    test_basic()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "summarize":
            test_summarization()
        elif sys.argv[1] == "pipeline":
            test_pipeline()
        elif sys.argv[1] == "all":
            test_summarization()
            test_pipeline()
    else:
        print("\n[INFO] Usage:")
        print("  python api_test_client.py summarize  - Test summarization")
        print("  python api_test_client.py pipeline   - Test transcribe+summarize")
        print("  python api_test_client.py all        - Test everything")
        print("\n[NOTE] Make sure API is running: python api/server.py")


if __name__ == "__main__":
    main()
