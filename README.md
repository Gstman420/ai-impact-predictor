# AI Impact Predictor 🔍

An AI-powered requirements analysis system that intelligently classifies, analyzes dependencies, detects impacts, and summarizes software requirements using state-of-the-art machine learning models.

## Features ✨

- **Requirement Classification**: Automatically categorize requirements (Functional, Non-Functional, Performance, Security, Usability)
- **Dependency Detection**: Find relationships between requirements using advanced embeddings
- **Impact Analysis**: Analyze cascading impacts of requirement changes using financial impact models
- **Smart Summarization**: Generate concise summaries of multiple requirements
- **Code Analysis**: Analyze code snippets and provide suggestions

## Tech Stack 🛠️

- **Backend**: FastAPI + Python
- **AI/ML**: PyTorch + Transformers (HuggingFace models)
- **Database**: SQLAlchemy + SQLite
- **File Processing**: Pandas, OpenPyXL
- **Frontend**: HTML/JavaScript (basic test interface included)

## Quick Start 🚀

### 1. Clone the Repository
```bash
git clone https://github.com/Gstman420/ai-impact-predictor.git
cd ai-impact-predictor
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Models
See [MODEL_SETUP.md](MODEL_SETUP.md) for detailed model setup instructions.

Models will auto-download on first run (requires ~25 GB storage and 20-40 min).

### 5. Run the Application
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 6. Test Frontend (Optional)
Open `test_frontend_connection.html` in your browser to test the API endpoints.

**Note**: Full frontend interface is under development and will be added in future updates.

## Project Structure 📁

```
ai-impact-predictor/
├── ai_engine/          # AI model loading logic
├── services/           # Business logic (AI service, impact analysis)
├── models/             # Data models and schemas
├── routes/             # API endpoints
├── database/           # Database files (not in git)
├── uploads/            # Uploaded files (not in git)
├── config.py           # Configuration
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── test_frontend_connection.html  # Basic frontend test interface
```

## API Endpoints 📡

- `POST /analyze/classify` - Classify requirements
- `POST /analyze/dependencies` - Find requirement dependencies
- `POST /analyze/impact` - Analyze impact of changes
- `POST /analyze/summarize` - Generate summaries
- `POST /analyze/batch` - Complete batch analysis

## AI Models Used 🤖

This project uses multiple pre-trained models from HuggingFace:
- **Requirement Classifier** (440 MB)
- **Qwen3 Embedder** (16 GB)
- **BART Summarizer** (1.6 GB)
- **Financial Impact Analyzer** (500 MB)
- **StarCoder2 Code Analyzer** (7 GB)

**Total Size**: ~25 GB

See [MODEL_SETUP.md](MODEL_SETUP.md) for complete details.

## System Requirements 💻

- **RAM**: 32 GB recommended (minimum 16 GB)
- **Storage**: 25+ GB for models
- **GPU**: CUDA GPU recommended (CPU fallback available)
- **Python**: 3.8+

## Configuration ⚙️

Edit `config.py` to customize:
- Upload directory
- Database location
- File size limits
- Similarity thresholds
- Impact analysis depth

## Development 👨‍💻

### Running Tests
```bash
python test_backend.py
```

### Code Style
```bash
black .
flake8 .
```

## Troubleshooting 🔧

**Out of Memory?**
- Reduce batch size
- Use CPU mode
- Close other applications

**Models not downloading?**
- Check internet connection
- Verify HuggingFace Hub access
- Try manual download

**Slow performance?**
- Enable GPU (CUDA)
- Reduce max impact depth
- Use smaller batch sizes

## Roadmap 🗺️

- [ ] Complete frontend interface with modern UI
- [ ] Real-time collaboration features
- [ ] Advanced visualization dashboards
- [ ] Model fine-tuning capabilities
- [ ] Docker containerization
- [ ] API authentication

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📄

MIT License (or specify your license)

## Acknowledgments 🙏

- HuggingFace for pre-trained models
- FastAPI framework
- PyTorch and Transformers library

## Contact 📧

GitHub: [@Gstman420](https://github.com/Gstman420)

---

**Note**: This project requires downloading ~25 GB of AI models. See [MODEL_SETUP.md](MODEL_SETUP.md) for setup instructions.