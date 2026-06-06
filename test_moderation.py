#!/usr/bin/env python3
"""
Test script for content moderation pipeline.
Tests Vietnamese, English, and mixed EN/VI text.
"""

import requests
import json
from typing import Dict, List

# API endpoint
BASE_URL = "http://127.0.0.1:8080"
MODERATION_ENDPOINT = f"{BASE_URL}/moderation/text"

# Test cases: (language, text, description)
TEST_CASES = [
    # ========== VIETNAMESE TEXT ==========
    (
        "VI",
        "Xin chào các bạn, tôi rất vui được gặp các bạn hôm nay. Tôi hy vọng chúng ta có một cuộc trò chuyện tuyệt vời và thoải mái với nhau. Cảm ơn các bạn đã dành thời gian để nói chuyện với tôi.",
        "VI - Safe greeting & appreciation"
    ),
    (
        "VI",
        "Bạn là một người ngu ngốc và vô dụng hoàn toàn. Tôi không hiểu làm sao mà những người như bạn lại tồn tại trên thế giới này. Bạn không xứng đáng được sống và làm việc cùng những người thông minh như tôi.",
        "VI - Insult & dehumanization (toxic)"
    ),
    (
        "VI",
        "Tôi không thích cách bạn hành động trong buổi họp hôm qua, nhưng tôi vẫn tôn trọng bạn như một con người. Chúng ta có thể cùng nhau cải thiện tình hình này và tìm ra một giải pháp tốt hơn cho cả hai bên.",
        "VI - Borderline (critical but respectful)"
    ),
    (
        "VI",
        "Địt mẹ mày, bạn là một con chó điên dại và không có trí tuệ gì. Tôi sẽ không bao giờ nói chuyện với những kẻ như mày nữa. Mất tinh thần, mất khách sáng tạo, mất tất cả những gì tốt đẹp.",
        "VI - Severe toxicity with curses (block)"
    ),
    
    # ========== ENGLISH TEXT ==========
    (
        "EN",
        "Hello everyone, I'm delighted to meet you all today. I really hope we can have a wonderful and comfortable conversation together. Thank you so much for taking the time to talk with me and share your thoughts.",
        "EN - Safe greeting & appreciation"
    ),
    (
        "EN",
        "You are absolutely stupid and worthless. I cannot understand how people like you even manage to exist in this world. You don't deserve to live or work alongside intelligent people like myself. Your existence is a mistake.",
        "EN - Insult & dehumanization (toxic)"
    ),
    (
        "EN",
        "I didn't like your behavior during yesterday's meeting, but I still respect you as a person. We can work together to improve this situation and find a better solution that works for both of us.",
        "EN - Borderline (critical but respectful)"
    ),
    (
        "EN",
        "Go fuck yourself you piece of shit, you're a complete idiot and a waste of space. Nobody wants you around and nobody cares what you think. You should just disappear and never come back again.",
        "EN - Severe toxicity with curses (block)"
    ),
    (
        "EN",
        "what is the smartest age perhaps a day of friendly competition will lead us to the answer tomorrow's the annual brain clash ten teams of two competing in a decathlon of mental challenges trivia competitions and puzzles i've been training all year i'll need to pick the smartest most capable teammate i've narrowed down the roster first we have gabriella now she may only be eight but don't underestimate her she's fluent in two languages and is the ultimate outside the box thinker then there's amma she can recite one hundred digits of pie designed satellites for a living and bakes a perfect souffle or i could go with mr taylor he's the best chess player in the neighborhood not to mention he's competed in over twenty brain clashes and is a five time champion i'm not sure who to pick who's the smartest which of these teammates should amir choose for tomorrow's contest and why of course it depends while intelligence is often associated with things like iq tests these assessments fail to capture the scope and depth of a person's varied abilities so instead we'll break down the idea of smart into categories like creativity memory and learning and explore when the brain's best at each of them let's start at the very beginning in the first few years of life your brain undergoes incredible rapid growth called synaptogenesis where more than one million new neural connections are formed every second as the brain develops it goes through a pruning process based on your experience and environment used connections are strengthened and unused connections are removed frequently used neuronal pathways are myelinated wrapped in a layer of insulation allowing information to travel faster this creates a more efficient fine tuned brain but this brain remodeling happens within and between brain regions at different times allowing different skills to flourish at different ages for example in childhood brain regions involved in language learning develop quickly which is why many children can learn and master multiple languages yet the prefrontal cortex a brain region responsible for cognitive control and inhibition is slower to develop as a result some young children may struggle with strategic games such as chess or checkers which require constant concentration planning and abstract thought at the same time children tend to be more flexible exploration based learners they often use more creative approaches when finding solutions to riddles and are on average less afraid to make mistakes but adults have their own unique set of abilities adults benefit from a well developed prefrontal cortex allowing them to better execute skills that require learning focus and memory making them quick and efficient puzzle solvers or crossword masters late in adulthood these same skills may decline as the brain's memory center known as the hippocampus shrinks but there's a reason for the phrase older and wiser after a lifetime of learning older adults have more knowledge to recall and utilize making them excellent trivia partners other factors that amir should consider are his own strengths as an adolescent the prefrontal cortical regions of your brain are more developed than in childhood this allows you to better navigate logic and math puzzles simultaneously deep inside the brain regions that are important in motivation and reward are developing even faster driving teenagers like amir to be curious and adventurous learners in many ways you can think of the teenager as a jack of all trades with brains wired to seek out new experiences and learn quickly you're at a dynamic stage where the choices you make and the skills you focus on can actually guide the development of your brain so what's the smartest age there's no single answer it's eight sixteen twenty five sixty five sixty five and everything in between our brains have adapted to prioritize different skills at various ages to meet that stage of life's challenges and demands so no matter who amir picks having an age diverse team is a good strategy that's right no matter your age you have ideas that are worth sharing that's why students around the world are honing their ideas into ted style talks with the ted ed student talks program teachers learn how to bring this free curriculum to a classroom or after school program near you at ed.ted.com slash student talks",
        "EN - Educational transcript about brain development (should be SAFE, no toxic content)"
    ),
    # ========== MIXED EN/VI TEXT ==========
    (
        "MIX",
        "Hello bạn, how are you doing today? Tôi rất happy được nhìn thấy bạn lại. I hope everything is going well with you and your family. Mọi việc của tôi đều tốt, thank you for asking.",
        "MIX - Safe mixed greeting & conversation"
    ),
    (
        "MIX",
        "You are stupid và bạn là một người ngu ngốc hoàn toàn. I hate people like you và tôi không thích những loại người như mày. You don't deserve respect và bạn không xứng đáng được sống.",
        "MIX - Mixed insult with toxicity"
    ),
    (
        "MIX",
        "Bạn sucks man, you're not my type at all. Your personality is annoying và tôi không muốn nói chuyện với bạn nữa. I don't think we have anything in common, và chúng ta maybe nên stop liên lạc.",
        "MIX - Mixed moderate criticism"
    ),
    (
        "MIX",
        "Địt mẹ mày fuck you, you're completely worthless and I hate every minute with you. Bạn là một con chó điên dại và tôi muốn bạn biến mất. Go away and never come back, tớ không cần thấy mặt mày bao giờ nữa.",
        "MIX - Mixed severe toxicity (block)"
    ),
    
    # ========== EDGE CASES ==========
    (
        "EDGE",
        "This is a simple test message to verify the moderation system works correctly with different kinds of content.",
        "Edge - Normal message"
    ),
    (
        "EDGE",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once, which makes it useful for testing purposes.",
        "Edge - Pangram test"
    ),
    (
        "EDGE",
        "Việt Nam là một đất nước xinh đẹp với những con người thân thiện và hiếu khách. Chúng tôi rất tự hào về lịch sử và văn hóa của đất nước mình.",
        "Edge - Positive Vietnamese content"
    ),
]


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(case_num: int, lang: str, text: str, description: str, result: Dict) -> None:
    """Pretty print test result."""
    print(f"\n[Test {case_num}] {lang} - {description}")
    print(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    
    if "moderation" in result:
        mod = result["moderation"]
        print(f"Status: {mod.get('status')} (score: {mod.get('score', 0.0):.3f})")
        print(f"Toxic words: {mod.get('toxic_word', [])}")
        print(f"Categories: {mod.get('categories', [])}")
    elif "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Response: {json.dumps(result, indent=2)}")


def test_moderation() -> None:
    """Run moderation tests."""
    print_header("MODERATION TESTS")
    print(f"API Endpoint: {MODERATION_ENDPOINT}")
    
    results_summary = {
        "safe": 0,
        "review": 0,
        "block": 0,
        "error": 0,
    }
    
    for idx, (lang, text, description) in enumerate(TEST_CASES, 1):
        try:
            # Skip empty text
            if not text.strip():
                print(f"\n[Test {idx}] {lang} - {description}")
                print("Skipped (empty text)")
                continue
            
            # Make request
            response = requests.post(
                MODERATION_ENDPOINT,
                json={
                    "text": text,
                },
                timeout=30,
            )
            
            print(f"\n[Test {idx}] {lang} - {description}")
            print(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Debug: print the full response on the first test.
                    if idx == 1:
                        print(f"[DEBUG] Full response structure:")
                        print(json.dumps(result, indent=2, ensure_ascii=False)[:800])
                    
                    if result.get("status") == "success":
                        moderation = result.get("moderation", {})
                        status = moderation.get("status", "ERROR")
                        score = moderation.get("score", 0)
                        
                        status_key = status.lower() if status else "error"
                        if status_key in results_summary:
                            results_summary[status_key] += 1
                        else:
                            results_summary["error"] += 1
                        
                        print(f"Status: {status} (score: {score:.3f})")
                        print(f"Toxic words: {moderation.get('toxic_word', [])}")
                        print(f"Categories: {moderation.get('categories', [])}")
                    else:
                        print(f"Response Status: {result.get('status')}")
                        print(f"Error: {result.get('error', 'Unknown')}")
                        print(f"Full response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
                        results_summary["error"] += 1
                except json.JSONDecodeError as e:
                    print(f"JSON Parse Error: {str(e)}")
                    print(f"Response text: {response.text[:200]}")
                    results_summary["error"] += 1
            else:
                print(f"HTTP Error: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                results_summary["error"] += 1
                
        except requests.exceptions.Timeout:
            print(f"\n[Test {idx}] {lang} - {description}")
            print("Timeout: Request took longer than 30 seconds")
            results_summary["error"] += 1
        except requests.exceptions.ConnectionError:
            print(f"\n[Test {idx}] {lang} - {description}")
            print(f"Connection Error: Cannot reach {BASE_URL}")
            print("Make sure API is running: python -m api.server")
            results_summary["error"] += 1
        except Exception as e:
            print(f"\n[Test {idx}] {lang} - {description}")
            print(f"Exception: {type(e).__name__}: {str(e)}")
            results_summary["error"] += 1
    
    # Print summary
    print_header("SUMMARY")
    total_run = sum(results_summary.values())
    print(f"Total tests run: {total_run}")
    print()
    print(f"✅ SAFE (Passed):   {results_summary['safe']:2d} tests")
    print(f"⚠️  REVIEW:         {results_summary['review']:2d} tests")
    print(f"🚫 BLOCK (Failed):  {results_summary['block']:2d} tests")
    print(f"❌ ERRORS:         {results_summary['error']:2d} tests")
    print()
    
    # Pass/Fail ratio
    passed = results_summary['safe']
    failed = results_summary['block']
    if total_run > 0:
        pass_rate = (passed / (total_run - results_summary['error'])) * 100 if (total_run - results_summary['error']) > 0 else 0
        print(f"Pass Rate: {passed}/{total_run - results_summary['error']} = {pass_rate:.1f}%")
        
        # Visual bar
        bar_length = 40
        filled = int(bar_length * passed / (total_run - results_summary['error'])) if (total_run - results_summary['error']) > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"[{bar}]")


def highlight_toxic_words(text: str, matched_spans: List[Dict]) -> str:
    """Highlight toxic words in text."""
    if not matched_spans:
        return text
    
    # Sort spans by position (longest first to avoid overlap issues)
    sorted_spans = sorted(matched_spans, key=lambda x: len(x["text"]), reverse=True)
    
    result = text
    for span in sorted_spans:
        span_text = span["text"]
        # Wrap toxic word with markers
        result = result.replace(span_text, f"[TOXIC: {span_text}]")
    
    return result


def print_toxic_analysis(moderation: Dict) -> None:
    """Print detailed toxic word analysis."""
    matched_spans = moderation.get("matched_spans", [])
    
    if not matched_spans:
        print("✅ No toxic words detected")
        return
    
    print(f"⚠️  Found {len(matched_spans)} toxic span(s):")
    print()
    
    for i, span in enumerate(matched_spans, 1):
        span_text = span.get("text", "")
        score = span.get("score", 0)
        categories = span.get("categories", [])
        lang = span.get("lang", "unknown")
        
        # Determine severity
        if score >= 0.85:
            severity = "🚫 BLOCK"
        elif score >= 0.55:
            severity = "⚠️  REVIEW"
        else:
            severity = "⚡ WARN"
        
        print(f"  {i}. {severity} | Score: {score:.3f} | Lang: {lang}")
        print(f"     Text: \"{span_text}\"")
        if categories:
            print(f"     Categories: {', '.join(categories)}")
        print()


def test_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print_header("API HEALTH CHECK")
            print(f"Status: ✅ Running")
            models = health.get('data', {}).get('models_loaded', [])
            print(f"Models loaded: {', '.join(models)}")
            return True
        else:
            print("❌ API returned non-200 status")
            print(f"Status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API Connection Error")
        print(f"Cannot reach {BASE_URL}")
        print("Start the API with: python -m api.server")
        return False
    except Exception as e:
        print(f"❌ API health check failed: {str(e)}")
        return False


def test_single(text: str) -> None:
    """Test a single text for debugging."""
    print_header("SINGLE TEXT TEST")
    print(f"Text: {text}")
    print(f"Endpoint: {MODERATION_ENDPOINT}\n")
    
    try:
        payload = {"text": text}
        print(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        response = requests.post(
            MODERATION_ENDPOINT,
            json=payload,
            timeout=30,
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"\nResponse Body:")
        print(response.text)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nParsed JSON (with Vietnamese):")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Detailed analysis
            if result.get("status") == "success":
                moderation = result.get("moderation", {})
                print("\n" + "="*80)
                print("DETAILED ANALYSIS")
                print("="*80)
                print(f"Label: {moderation.get('label')} (Score: {moderation.get('score'):.3f})")
                print()
                print_toxic_analysis(moderation)
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    import sys
    
    # Check API health first
    if not test_api_health():
        exit(1)
    
    # Check for debug mode
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        text = sys.argv[2] if len(sys.argv) > 2 else "This is a test"
        test_single(text)
    else:
        test_moderation()
        
