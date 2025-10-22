\# Model Setup Guide



This project uses 5 pretrained models from HuggingFace. Models are NOT included in the repository due to their large size.



\## Models Used



\### 1. \*\*Requirement Classifier\*\*

\- \*\*Model\*\*: `rajinikarcg/RequirementClassifier`

\- \*\*Purpose\*\*: Classifies requirements into categories (Functional, Non-Functional, Performance, Security, Usability)

\- \*\*Size\*\*: ~440 MB



\### 2. \*\*Embedder\*\*

\- \*\*Model\*\*: `Qwen/Qwen3-Embedding-8B`

\- \*\*Purpose\*\*: Generates embeddings for similarity analysis and dependency detection

\- \*\*Size\*\*: ~16 GB



\### 3. \*\*Summarizer\*\*

\- \*\*Model\*\*: `facebook/bart-large-cnn`

\- \*\*Purpose\*\*: Generates summaries of multiple requirements

\- \*\*Size\*\*: ~1.6 GB



\### 4. \*\*Financial Impact Analyzer\*\*

\- \*\*Model\*\*: `nusret35/roberta-financial-news-impact-analysis`

\- \*\*Purpose\*\*: Analyzes business/financial impact of requirement changes

\- \*\*Size\*\*: ~500 MB



\### 5. \*\*Code Analyzer\*\*

\- \*\*Model\*\*: `bigcode/starcoder2-7b`

\- \*\*Purpose\*\*: Analyzes code snippets and provides suggestions

\- \*\*Size\*\*: ~7 GB



\*\*Total Storage Required\*\*: ~25 GB



\## Setup Instructions



\### Option 1: Automatic Download (Recommended)



The models will automatically download from HuggingFace on first use. This may take 20-40 minutes depending on your internet speed.



1\. \*\*Install dependencies\*\*:

```bash

&nbsp;  pip install -r requirements.txt

```



2\. \*\*Run the application\*\*:

```bash

&nbsp;  python main.py

```



3\. \*\*Models will auto-download\*\* to your cache directory:

&nbsp;  - Windows: `C:\\Users\\<username>\\.cache\\huggingface\\`

&nbsp;  - Linux/Mac: `~/.cache/huggingface/`



\### Option 2: Pre-download Models



Download models individually before running the application:

```bash

\# Download BART summarizer

python download\_better\_summarizer.py



\# Download StarCoder2 code analyzer

python download\_starcoder.py



\# Download all remaining models

python setup\_models\_cache.py

```



\### Option 3: Manual Download



If you already have models cached elsewhere:



1\. Download models from HuggingFace Hub

2\. Place them in your HuggingFace cache directory

3\. Or modify `model\_manager.py` to point to your model directory



\## System Requirements



\- \*\*RAM\*\*: 32 GB recommended (minimum 16 GB)

\- \*\*Storage\*\*: 25+ GB free space for all models

\- \*\*GPU\*\*: CUDA-compatible GPU strongly recommended (CPU fallback available)

\- \*\*Python\*\*: 3.8+

\- \*\*Internet\*\*: Required for initial model download



\## Cache Location



Models are cached in:

\- \*\*Windows\*\*: `C:\\Users\\<YourUsername>\\.cache\\huggingface\\hub\\`

\- \*\*Linux\*\*: `~/.cache/huggingface/hub/`

\- \*\*Mac\*\*: `~/.cache/huggingface/hub/`



\## Model Details



| Model | Type | Size | Use Case |

|-------|------|------|----------|

| RequirementClassifier | BERT-based | 440 MB | Text Classification |

| Qwen3-Embedding-8B | Embedding | 16 GB | Semantic Similarity |

| BART-Large-CNN | Seq2Seq | 1.6 GB | Text Summarization |

| RoBERTa-Financial | BERT-based | 500 MB | Sentiment Analysis |

| StarCoder2-7B | Code LLM | 7 GB | Code Analysis |



\## Troubleshooting



\### Out of Memory

\- Reduce batch size in config

\- Use CPU mode for non-critical models

\- Close other applications

\- Process requirements in smaller batches



\### Download Fails

\- Check internet connection

\- Verify HuggingFace Hub access (some models may need authentication)

\- Try downloading one model at a time

\- Check disk space



\### Model Not Found / Load Error

\- Clear cache: Delete `~/.cache/huggingface/` and re-download

\- Check model names in `model\_manager.py`

\- Verify all dependencies are installed

\- Ensure Python version compatibility



\### Slow Performance

\- Enable GPU (CUDA) if available

\- Reduce `max\_length` parameters

\- Use smaller batch sizes

\- Consider using quantized models (future enhancement)



\## Performance Notes



\- \*\*First Run\*\*: 20-40 minutes (model download + initialization)

\- \*\*Subsequent Runs\*\*: 2-5 minutes (model loading only)

\- \*\*GPU Usage\*\*: Models automatically use GPU if CUDA is available

\- \*\*CPU Fallback\*\*: All models work on CPU but 3-5x slower

\- \*\*Memory Usage\*\*: Peak usage ~20 GB RAM with all models loaded



\## Development Notes



\- Models use `local\_files\_only=True` flag after initial download

\- Lazy loading: Models load only when needed

\- Singleton pattern: Each model loads once per session

\- Thread-safe: Can handle concurrent requests



\## HuggingFace Authentication



Some models may require HuggingFace authentication:

```bash

\# Login to HuggingFace (one-time setup)

huggingface-cli login

```



Enter your HuggingFace token when prompted.

