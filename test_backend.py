"""
Test script to verify all 5 AI models are working
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_health_check():
    """Test 1: Health Check"""
    print_section("TEST 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    assert response.status_code == 200, "Health check failed!"
    print("✅ Health check passed!")

def test_model_status():
    """Test 2: Check Model Status"""
    print_section("TEST 2: Model Status")
    
    response = requests.get(f"{BASE_URL}/api/models")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    assert response.status_code == 200, "Model status check failed!"
    print("✅ Model status check passed!")

def test_upload_requirements():
    """Test 3: Upload Sample Requirements"""
    print_section("TEST 3: Upload Requirements")
    
    # Create sample requirements file
    sample_requirements = """The system shall authenticate users using email and password
The system shall respond to user requests within 2 seconds
The system shall encrypt all sensitive data using AES-256
The system shall support at least 1000 concurrent users
The system shall have 99.9% uptime availability"""
    
    # Upload file
    files = {
        'file': ('test_requirements.txt', sample_requirements, 'text/plain')
    }
    
    response = requests.post(f"{BASE_URL}/api/upload", files=files)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    assert response.status_code == 200, "Upload failed!"
    print("✅ Upload test passed!")

def test_classify_requirements():
    """Test 4: Classify Requirements (Model 1)"""
    print_section("TEST 4: Classify Requirements (Model 1: Classifier)")
    
    response = requests.post(f"{BASE_URL}/api/classify")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total classified: {data['total']}")
        
        # Show first 2 classifications
        for i, req in enumerate(data['classifications'][:2], 1):
            print(f"\n  Requirement {i}:")
            print(f"    Text: {req['text'][:60]}...")
            print(f"    Category: {req['ai_category']}")
            print(f"    Confidence: {req['confidence']:.2f}")
        
        print("✅ Classification test passed!")
    else:
        print(f"❌ Classification failed: {response.text}")

def test_find_dependencies():
    """Test 5: Find Dependencies (Model 2)"""
    print_section("TEST 5: Find Dependencies (Model 2: Embedder)")
    
    response = requests.post(
        f"{BASE_URL}/api/dependencies",
        params={"threshold": 0.5}
    )
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total dependencies found: {data['total']}")
        
        # Show first 2 dependencies
        for i, dep in enumerate(data['dependencies'][:2], 1):
            print(f"\n  Dependency {i}:")
            print(f"    {dep['source_id']} ↔ {dep['target_id']}")
            print(f"    Similarity: {dep['similarity_score']:.3f}")
            print(f"    Type: {dep['dependency_type']}")
        
        print("✅ Dependency test passed!")
    else:
        print(f"❌ Dependency analysis failed: {response.text}")

def test_summarize():
    """Test 6: Summarize Requirements (Model 3)"""
    print_section("TEST 6: Summarize Requirements (Model 3: Summarizer)")
    
    response = requests.post(f"{BASE_URL}/api/summarize")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Summary: {data['summary'][:200]}...")
        print(f"Compression ratio: {data['compression_ratio']}x")
        print("✅ Summarization test passed!")
    else:
        print(f"❌ Summarization failed: {response.text}")

def test_impact_analysis():
    """Test 7: Impact Analysis (Model 4)"""
    print_section("TEST 7: Impact Analysis (Model 4: Impact Analyzer)")
    
    # First get a requirement ID
    req_response = requests.get(f"{BASE_URL}/api/requirements")
    if req_response.status_code == 200:
        reqs = req_response.json()['requirements']
        if reqs:
            req_id = reqs[0]['req_id']
            
            response = requests.post(
                f"{BASE_URL}/api/impact",
                json={"changed_req_id": req_id, "threshold": 0.5}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Changed requirement: {data['changed_requirement']}")
                print(f"Impact sentiment: {data.get('impact_sentiment', 'N/A')}")
                print(f"Total impacted: {data['total_impacted']}")
                print(f"Risk level: {data['overall_risk_level']}")
                print("✅ Impact analysis test passed!")
            else:
                print(f"❌ Impact analysis failed: {response.text}")
    else:
        print("❌ Could not get requirements for impact test")

def test_code_analysis():
    """Test 8: Code Analysis (Model 5)"""
    print_section("TEST 8: Code Analysis (Model 5: Code Analyzer)")
    
    sample_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item
    return total
"""
    
    response = requests.post(
        f"{BASE_URL}/api/analyze-code",
        json={"code": sample_code}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Analysis: {data['analysis'][:200]}...")
        print(f"Suggestions: {len(data['suggestions'])} provided")
        print("✅ Code analysis test passed!")
    else:
        print(f"❌ Code analysis failed: {response.text}")

def test_batch_analysis():
    """Test 9: Batch Analysis (Models 1, 2, 3)"""
    print_section("TEST 9: Batch Analysis (Models 1, 2, 3)")
    
    response = requests.post(
        f"{BASE_URL}/api/analyze",
        params={"threshold": 0.5}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total requirements analyzed: {data['total_requirements']}")
        print(f"Dependencies found: {data['dependencies']['total']}")
        print(f"Summary length: {len(data['summary']['summary'])} chars")
        print("✅ Batch analysis test passed!")
    else:
        print(f"❌ Batch analysis failed: {response.text}")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("  🚀 TESTING ALL 5 AI MODELS")
    print("="*60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Status", test_model_status),
        ("Upload Requirements", test_upload_requirements),
        ("Classify (Model 1)", test_classify_requirements),
        ("Dependencies (Model 2)", test_find_dependencies),
        ("Summarize (Model 3)", test_summarize),
        ("Impact Analysis (Model 4)", test_impact_analysis),
        ("Code Analysis (Model 5)", test_code_analysis),
        ("Batch Analysis", test_batch_analysis),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            time.sleep(1)  # Wait between tests
        except Exception as e:
            print(f"❌ {name} FAILED: {str(e)}")
            failed += 1
    
    print_section("TEST SUMMARY")
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Your backend is ready!")
    else:
        print(f"\n⚠️ {failed} tests failed. Check the errors above.")

if __name__ == "__main__":
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server!")
        print("   Make sure your FastAPI server is running:")
        print("   python main.py")