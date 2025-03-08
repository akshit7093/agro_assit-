# Multi-Functional AI Assistant

This repository contains a multi-functional AI assistant leveraging Large Language Models (LLMs) to provide advanced natural language processing capabilities.

## 🚀 Features

- **LLM Integration & Abstraction**: Supports multiple LLM providers with an abstraction layer.
- **Data Handling**:
  - Ingests and processes CSV, JSON, and plain text files.
- **Core NLP Functionalities**:
  - Text Summarization
  - Sentiment Analysis
  - Named Entity Recognition (NER)
  - Question Answering
  - Code Generation & Assistance
- **Retrieval-Augmented Generation (RAG)**:
  - Document indexing and semantic search.
- **Conversational Interface**:
  - Multi-turn dialogue with persona switching.
- **Performance Optimization**:
  - Caching, prompt engineering, and monitoring.

## 📁 Project Structure

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

## 🛠️ Setup & Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/yourusername/multi-functional-ai-assistant.git
   cd multi-functional-ai-assistant
   ```
2. **Create a Virtual Environment:**
   ```sh
   python -m venv venv
   ```
3. **Activate the Virtual Environment:**
   - **Windows:** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`
4. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
5. **Set Up Configuration:**
   - Copy `config/config.example.yaml` to `config/config.yaml` and update it with your settings.
   - Create a `.env` file inside the `config/` directory and add necessary API keys:
     ```
     OPENROUTER_API_KEY=your_openrouter_api_key
     ```
6. **Run the Application:**
   ```sh
   streamlit run ui/streamlit_app.py
   ```

## 🤖 Supported Models

The AI assistant integrates models via OpenRouter:

- **Google:** Gemini Flash Lite 2.0 Preview (free)
- **DeepSeek:** R1 (free)
- **Qwen:** Qwen2.5 VL 72B Instruct (free)
- **NVIDIA:** Llama 3.1 Nemotron 70B Instruct (free)

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 💡 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push the branch (`git push origin feature-branch`).
5. Open a pull request.

## 📞 Contact

For any inquiries, feel free to reach out via GitHub Issues or email at `your.email@example.com`.

---

🌟 If you like this project, consider giving it a star! ⭐

