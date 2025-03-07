<<<<<<< HEAD
<<<<<<< HEAD
# Multi-Functional AI Assistant

This project implements a multi-functional AI assistant leveraging large language models (LLMs) with the following capabilities:

## Features

- **LLM Integration and Abstraction**: Supports multiple LLM providers with an abstraction layer
- **Data Ingestion and Processing**: Handles CSV, JSON, and plain text files
- **Core NLP Functionalities**:
  - Text Summarization
  - Sentiment Analysis
  - Named Entity Recognition (NER)
  - Question Answering
  - Code Generation & Assistance
- **Retrieval-Augmented Generation (RAG)**: Document indexing and semantic search
- **Conversational Interface**: Multi-turn dialogue with persona switching
- **Performance Optimization**: Caching, prompt engineering, and monitoring

## Project Structure

```
├── config/                  # Configuration files
│   ├── config.yaml         # Main configuration
│   └── .env                # Environment variables (not tracked in git)
├── data/                   # Data storage
│   ├── raw/                # Raw input data
│   └── processed/          # Processed data
├── models/                 # Model abstraction and implementations
│   ├── base.py            # Base model interface
│   ├── openrouter.py      # OpenRouter implementation
│   └── factory.py         # Model factory
├── processors/            # Data processing modules
│   ├── ingestion.py       # Data ingestion
│   ├── preprocessing.py   # Text preprocessing
│   └── utils.py           # Utility functions
├── nlp/                   # NLP functionality modules
│   ├── summarization.py   # Text summarization
│   ├── sentiment.py       # Sentiment analysis
│   ├── ner.py             # Named Entity Recognition
│   ├── qa.py              # Question answering
│   └── code_assistant.py  # Code generation and assistance
├── rag/                   # Retrieval-Augmented Generation
│   ├── document_store.py  # Document storage and indexing
│   ├── retriever.py       # Document retrieval
│   └── prompt_builder.py  # Dynamic prompt construction
├── conversation/          # Conversation management
│   ├── session.py         # Session management
│   ├── context.py         # Context tracking
│   └── persona.py         # Persona management
├── ui/                    # User interface
│   ├── streamlit_app.py   # Streamlit application
│   └── components/        # UI components
├── utils/                 # Utility modules
│   ├── cache.py           # Caching mechanisms
│   ├── logger.py          # Logging utilities
│   └── metrics.py         # Performance metrics
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── app.py                 # Main application entry point
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup and Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file in the config directory with your API keys
6. Run the application: `streamlit run ui/streamlit_app.py`

## Configuration

Copy the `config/config.example.yaml` to `config/config.yaml` and update with your settings.

Create a `.env` file in the config directory with the following variables:

```
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Supported Models

Currently integrated models from OpenRouter:

- Google: Gemini Flash Lite 2.0 Preview (free)
- DeepSeek: R1 (free)
- Qwen: Qwen2.5 VL 72B Instruct (free)
- NVIDIA: Llama 3.1 Nemotron 70B Instruct (free)

## License

MIT
=======
# agro_assit-
>>>>>>> 03017cdc5a9ff389db5251d38cb06342c492602d
=======
# agro_assit-
>>>>>>> 03017cdc5a9ff389db5251d38cb06342c492602d
