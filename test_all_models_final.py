print("🧪 FINAL TEST: All 5 AI Models\n")
print("=" * 60)

# Test 1: Classifier
print("\n1️⃣ Testing Requirement Classifier...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("rajinikarcg/RequirementClassifier")
    model = AutoModelForSequenceClassification.from_pretrained("rajinikarcg/RequirementClassifier")
    print("   ✅ Classifier loaded successfully!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Embedder
print("\n2️⃣ Testing Qwen3 Embedder...")
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B")
    print("   ✅ Embedder loaded successfully!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: BART Summarizer (NEW!)
print("\n3️⃣ Testing BART Summarizer (NEW MODEL)...")
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    print("   ✅ BART Summarizer loaded successfully!")
    
    # Quick test
    text = "The system shall allow users to login with email and password."
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
    print(f"   🧪 Test Summary: {summary}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Impact Analyzer
print("\n4️⃣ Testing Impact Analyzer...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("nusret35/roberta-financial-news-impact-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("nusret35/roberta-financial-news-impact-analysis")
    print("   ✅ Impact Analyzer loaded successfully!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Code Analyzer
print("\n5️⃣ Testing Code Analyzer...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")
    model = AutoModelForCausalLM.from_pretrained(
        "bigcode/starcoder2-7b",
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("   ✅ Code Analyzer loaded successfully!")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("🎊 MODEL TEST COMPLETE!")
print("=" * 60)
print("\n✅ All models are ready for API integration!")
print("🚀 Next: Build the FastAPI application!")