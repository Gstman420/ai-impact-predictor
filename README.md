\# Requirements Analyzer 🔍



An AI-powered requirements analysis system that uses 5 state-of-the-art HuggingFace models to classify, analyze dependencies, detect impacts, and summarize software requirements.



\## Features ✨



\- \*\*Requirement Classification\*\*: Automatically categorize requirements (Functional, Non-Functional, Performance, Security, Usability)

\- \*\*Dependency Detection\*\*: Find relationships between requirements using advanced embeddings

\- \*\*Impact Analysis\*\*: Analyze cascading impacts of requirement changes using financial impact models

\- \*\*Smart Summarization\*\*: Generate concise summaries of multiple requirements

\- \*\*Code Analysis\*\*: Analyze code snippets (if applicable)



\## Tech Stack 🛠️



\- \*\*Backend\*\*: FastAPI + Python

\- \*\*AI/ML\*\*: PyTorch + Transformers (5 HuggingFace models)

\- \*\*Database\*\*: SQLAlchemy + SQLite

\- \*\*File Processing\*\*: Pandas, OpenPyXL



\## Quick Start 🚀



\### 1. Clone the Repository

```bash

git clone <your-repo-url>

cd requirements-analyzer

```



\### 2. Create Virtual Environment

```bash

python -m venv venv



\# Windows

venv\\Scripts\\activate



\# Linux/Mac

source venv/bin/activate

```



\### 3. Install Dependencies

```bash

pip install -r requirements.txt

```



\### 4. Setup Models

See \[MODEL\_SETUP.md](MODEL\_SETUP.md) for detailed model setup instructions.



Models will auto-download on first run (requires ~20 GB storage and 15-30 min).



\### 5. Run the Application

```bash

python main.py

```



The API will be available at `http://localhost:8000`



\## Project Structure 📁



```

requirements-analyzer/

├── ai\_engine/          # AI model loading logic

├── services/           # Business logic (AI service, impact analysis)

├── models/             # Data models and schemas

├── routes/             # API endpoints

├── database/           # Database files (not in git)

├── uploads/            # Uploaded files (not in git)

├── config.py           # Configuration

├── main.py             # Application entry point

└── requirements.txt    # Python dependencies

```



\## API Endpoints 📡



\- `POST /analyze/classify` - Classify requirements

\- `POST /analyze/dependencies` - Find requirement dependencies

\- `POST /analyze/impact` - Analyze impact of changes

\- `POST /analyze/summarize` - Generate summaries

\- `POST /analyze/batch` - Complete batch analysis



\## Models Used 🤖



This project uses 5 pre-trained models:

1\. \*\*Requirement Classifier\*\* (440 MB)

2\. \*\*Qwen3 Embedder\*\* (16 GB)

3\. \*\*M365 Summarizer\*\* (1.6 GB)

4\. \*\*Financial Impact Analyzer\*\* (500 MB)

5\. \*\*Code Analyzer\*\* (Variable)



\*\*Total Size\*\*: ~20 GB



See \[MODEL\_SETUP.md](MODEL\_SETUP.md) for details.



\## System Requirements 💻



\- \*\*RAM\*\*: 32 GB recommended (minimum 16 GB)

\- \*\*Storage\*\*: 20+ GB for models

\- \*\*GPU\*\*: CUDA GPU recommended (CPU fallback available)

\- \*\*Python\*\*: 3.8+



\## Configuration ⚙️



Edit `config.py` to customize:

\- Upload directory

\- Database location

\- File size limits

\- Similarity thresholds

\- Impact analysis depth



\## Development 👨‍💻



\### Running Tests

```bash

python test\_backend.py

```



\### Code Style

```bash

black .

flake8 .

```



\## Troubleshooting 🔧



\*\*Out of Memory?\*\*

\- Reduce batch size

\- Use CPU mode

\- Close other applications



\*\*Models not downloading?\*\*

\- Check internet connection

\- Verify HuggingFace Hub access

\- Try manual download



\*\*Slow performance?\*\*

\- Enable GPU (CUDA)

\- Reduce max impact depth

\- Use smaller batch sizes



\## Contributing 🤝



1\. Fork the repository

2\. Create a feature branch

3\. Commit your changes

4\. Push to the branch

5\. Open a Pull Request



\## License 📄



\[Your License Here]



\## Acknowledgments 🙏



\- HuggingFace for pre-trained models

\- FastAPI framework

\- PyTorch and Transformers library



\## Contact 📧



\[Your contact information]



---



\*\*Note\*\*: This project requires downloading ~20 GB of AI models. See MODEL\_SETUP.md for setup instructions.

