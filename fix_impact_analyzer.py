print("🔧 Fixing Impact Analyzer...\n")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "nusret35/roberta-financial-news-impact-analysis"
    
    print("📥 Re-downloading Impact Analyzer with force...")
    
    # Force clean download
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        force_download=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        force_download=True
    )
    
    print("✅ Impact Analyzer fixed and loaded!")
    
    # Quick test
    text = "The new feature will significantly improve user experience"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    
    print(f"🧪 Test successful! Output shape: {outputs.logits.shape}")
    print("\n🎉 Impact Analyzer is ready!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Alternative: We can proceed without this model")
    print("   The other 4 models are enough to build a powerful API!")