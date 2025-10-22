print("🔄 Downloading Better Impact Analyzer...\n")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # Using FinBERT - Industry standard for sentiment/impact analysis
    model_name = "ProsusAI/finbert"
    
    print(f"📥 Downloading: {model_name}")
    print("⚠️ Size: ~440MB\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✅ Tokenizer downloaded!")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("   ✅ Model downloaded!")
    
    # Quick test
    text = "The new system will significantly improve processing speed and reduce costs"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    
    import torch
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ['negative', 'neutral', 'positive']
    prediction = labels[probs.argmax()]
    
    print(f"\n🧪 Test Analysis:")
    print(f"   Text: {text}")
    print(f"   Impact: {prediction.upper()}")
    print(f"   Confidence: {probs.max().item():.2%}")
    
    print("\n🎉 FinBERT Impact Analyzer Ready!")
    print("✅ This model is actually BETTER than the broken one!")
    
except Exception as e:
    print(f"❌ Error: {e}")