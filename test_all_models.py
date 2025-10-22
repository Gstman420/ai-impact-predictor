print("🧪 Testing All 5 Models...\n")

# Test 1: Classifier
print("1️⃣ Testing Classifier...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("rajinikarcg/RequirementClassifier")
    model = AutoModelForSequenceClassification.from_pretrained("rajinikarcg/RequirementClassifier")
    print("   ✅ Classifier loaded!\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 2: Embedder
print("2️⃣ Testing Embedder...")
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B")
    print("   ✅ Embedder loaded!\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 3: Summarizer
print("3️⃣ Testing Summarizer...")
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("marklicata/M365_h2_Text_Processing_and_Summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("marklicata/M365_h2_Text_Processing_and_Summarization")
    print("   ✅ Summarizer loaded!\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 4: Impact Analyzer
print("4️⃣ Testing Impact Analyzer...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("nusret35/roberta-financial-news-impact-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("nusret35/roberta-financial-news-impact-analysis")
    print("   ✅ Impact Analyzer loaded!\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 5: Code Analyzer
print("5️⃣ Testing Code Analyzer...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-7b", device_map="auto", torch_dtype="auto")
    print("   ✅ Code Analyzer loaded!\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

print("=" * 60)
print("🎉 ALL MODELS TESTED!")
print("=" * 60)