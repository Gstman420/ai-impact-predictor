from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import upload, analysis
from database.connection import init_db
from models.model_manager import model_manager

app = FastAPI(
    title="Requirements Analysis API",
    description="AI-powered requirements dependency and impact analysis with 5 specialized models",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and AI models on startup
@app.on_event("startup")
async def startup_event():
    # Initialize database
    init_db()
    print("✅ Database initialized and ready!")
    
    # Load all AI models
    print("\n🤖 Initializing AI Models...")
    try:
        model_manager.load_all_models()
        print("✅ All AI models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Some models failed to load: {e}")

# Include routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])

@app.get("/")
def home():
    return {
        "message": "Requirements Analyzer Backend is Running!",
        "version": "2.0.0",
        "ai_models": model_manager.get_model_status(),
        "endpoints": {
            "upload": "/api/upload",
            "analysis": "/api/analysis",
            "models": "/api/models",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    model_status = model_manager.get_model_status()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "database": "connected",
        "ai_models": model_status
    }

@app.get("/api/models")
def model_info():
    """Get detailed information about loaded AI models"""
    return {
        "models": {
            "classifier": {
                "name": "RequirementClassifier",
                "purpose": "Classify requirements as Functional/Non-Functional",
                "status": "✅" if model_manager.get_model_status()["classifier"] else "❌"
            },
            "embedder": {
                "name": "Qwen3-Embedding-8B",
                "purpose": "Generate semantic embeddings for similarity analysis",
                "status": "✅" if model_manager.get_model_status()["embedder"] else "❌"
            },
            "summarizer": {
                "name": "BART-Large-CNN",
                "purpose": "Summarize long requirement documents",
                "status": "✅" if model_manager.get_model_status()["summarizer"] else "❌"
            },
            "impact_analyzer": {
                "name": "FinBERT",
                "purpose": "Analyze impact/sentiment of requirements",
                "status": "✅" if model_manager.get_model_status()["impact_analyzer"] else "❌"
            },
            "code_analyzer": {
                "name": "StarCoder2-7B",
                "purpose": "Analyze code and provide suggestions",
                "status": "✅" if model_manager.get_model_status()["code_analyzer"] else "❌"
            }
        },
        "device": model_manager.device,
        "total_loaded": model_manager.get_model_status()["total_loaded"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)