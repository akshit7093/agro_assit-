# Multi-Functional AI Assistant

A comprehensive AI assistant leveraging Large Language Models (LLMs) to provide advanced natural language processing capabilities with a modular architecture and extensive functionality.

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage Guide](#usage-guide)
- [Technical Implementation](#technical-implementation)
- [Supported Models](#supported-models)
- [Performance Optimization](#performance-optimization)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## 🔍 Overview

This Multi-Functional AI Assistant is designed to provide a versatile platform for natural language processing tasks using state-of-the-art LLMs. The system features a modular architecture that separates concerns and allows for flexibility and extensibility, making it suitable for a wide range of applications from simple text analysis to complex conversational interactions with document-grounded responses.

## 🚀 Features

- **LLM Integration & Abstraction**
  - Unified interface to multiple LLM providers
  - Model-agnostic architecture with factory pattern implementation
  - Configurable model parameters and response formats

- **Data Handling**
  - Ingests and processes CSV, JSON, and plain text files
  - Structured data parsing and transformation
  - Preprocessing pipeline with customizable steps

- **Core NLP Functionalities**
  - Text Summarization: Generate concise summaries of long documents
  - Sentiment Analysis: Determine emotional tone of text
  - Named Entity Recognition (NER): Extract people, organizations, locations, etc.
  - Question Answering: Provide accurate answers to natural language questions
  - Code Generation & Assistance: Generate and explain code in multiple languages

- **Retrieval-Augmented Generation (RAG)**
  - Document indexing with vector embeddings
  - Semantic search capabilities
  - Context-aware response generation
  - Hybrid retrieval strategies

- **Conversational Interface**
  - Multi-turn dialogue management
  - Context tracking across conversation turns
  - Persona switching for different use cases
  - Memory management for extended conversations

- **Performance Optimization**
  - Response caching for frequently asked questions
  - Prompt engineering techniques for better results
  - Performance monitoring and analytics
  - Configurable parameters for speed/quality trade-offs

## 🏗️ Architecture

The system is built with a modular architecture that separates concerns:

1. **Core Assistant (app.py)**: Central orchestrator that coordinates all functionalities
2. **Model Abstraction Layer**: Provides a unified interface to different LLM providers
3. **NLP Components**: Specialized modules for different NLP tasks
4. **Document Store**: Vector-based retrieval system for RAG capabilities
5. **Session Management**: Maintains conversation context and history
6. **UI Layer**: Streamlit-based interface for user interaction

### Key Design Patterns

- **Factory Pattern**: Used for model instantiation and configuration
- **Strategy Pattern**: Applied for swappable NLP processing techniques
- **Repository Pattern**: Implemented for document storage and retrieval
- **Facade Pattern**: Simplifies complex subsystem interactions

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

### Prerequisites

- Python 3.9+ installed
- Git for version control
- API keys for LLM providers (if using external models)

### Installation Steps

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

6. **Prepare Data Directory (Optional):**
   ```sh
   mkdir -p data/raw data/processed
   ```

7. **Run the Application:**
   ```sh
   streamlit run ui/streamlit_app.py
   ```

## 📖 Usage Guide

### Basic Usage

1. **Start the Application:**
   ```sh
   streamlit run ui/streamlit_app.py
   ```

2. **Access the Web Interface:**
   Open your browser and navigate to `http://localhost:8501`

3. **Select Functionality:**
   Choose from the sidebar which NLP function you want to use

### Using NLP Features

#### Text Summarization

```python
# Example code for using the summarization module
from nlp.summarization import Summarizer
from models.factory import ModelFactory

model_factory = ModelFactory()
summarizer = Summarizer(model_factory)

text = "Your long text here..."
result = await summarizer.summarize(text)
print(result['summary'])
```

#### Question Answering with RAG

```python
# Example code for using the QA module with RAG
from nlp.qa import QuestionAnswerer
from models.factory import ModelFactory
from rag.document_store import DocumentStore
from rag.retriever import DocumentRetriever

model_factory = ModelFactory()
document_store = DocumentStore()
retriever = DocumentRetriever(document_store)

# Add documents to the store
document_store.add_documents(["document1.txt", "document2.txt"])

# Initialize QA system
qa = QuestionAnswerer(model_factory)

# Ask a question
question = "What is the capital of France?"
relevant_docs = retriever.retrieve(question, top_k=3)
result = await qa.answer_question(question, context=relevant_docs)
print(result['answer'])
```

### Conversation Interface

The assistant provides a chat interface for multi-turn conversations:

```python
# Example code for using the conversation interface
from app import AIAssistant

assistant = AIAssistant()
response = await assistant.chat("Tell me about machine learning")
print(response)

# Continue the conversation
follow_up = await assistant.chat("What are some popular algorithms?")
print(follow_up)
```

## 🔧 Technical Implementation

### Model Abstraction Layer

The system uses a factory pattern to abstract away the specifics of different LLM providers:

```python
# models/factory.py
class ModelFactory:
    def get_model(self, model_name=None):
        if not model_name or model_name == "gemini":
            return OpenRouterModel(model="google/gemini-flash-lite-2.0-preview")
        elif model_name == "deepseek":
            return OpenRouterModel(model="deepseek/r1")
        # ... other models
```

### RAG Implementation

The Retrieval-Augmented Generation system works as follows:

1. **Document Indexing**: Documents are processed and stored as vector embeddings
2. **Query Processing**: User queries are converted to the same vector space
3. **Retrieval**: Semantic search finds the most relevant documents
4. **Context Integration**: Retrieved documents are incorporated into the prompt
5. **Response Generation**: The LLM generates a response based on the augmented context

```python
# Simplified RAG flow
query = "How does RAG work?"
relevant_docs = retriever.retrieve(query, top_k=3)
augmented_prompt = f"Answer based on these documents: {relevant_docs}\n\nQuestion: {query}"
response = await model.generate(augmented_prompt)
```

### Preprocessing Pipeline

The system includes a customizable preprocessing pipeline for text data:

1. **Text Cleaning**: Remove unwanted characters, normalize whitespace
2. **Tokenization**: Split text into tokens
3. **Normalization**: Convert to lowercase, remove accents
4. **Stop Word Removal**: Filter out common words with low information value
5. **Stemming/Lemmatization**: Reduce words to their root forms

## 🤖 Supported Models

The AI assistant integrates models via OpenRouter:

- **Google:** Gemini Flash Lite 2.0 Preview (free)
  - Optimized for fast responses and general-purpose tasks
  - Good balance of performance and speed

- **DeepSeek:** R1 (free)
  - Specialized in reasoning and problem-solving
  - Strong performance on complex analytical tasks

- **Qwen:** Qwen2.5 VL 72B Instruct (free)
  - Vision-language model with multimodal capabilities
  - Excellent for tasks involving both text and images

- **NVIDIA:** Llama 3.1 Nemotron 70B Instruct (free)
  - Large context window and strong reasoning abilities
  - Performs well on complex, multi-step tasks

## ⚡ Performance Optimization

### Caching Strategy

The system implements a multi-level caching strategy:

1. **Response Caching**: Frequently asked questions are cached
2. **Embedding Caching**: Document embeddings are cached to avoid recomputation
3. **Model Output Caching**: Similar prompts may return cached responses

### Prompt Engineering

Carefully crafted prompts improve model performance:

1. **Task-Specific Templates**: Optimized prompts for different NLP tasks
2. **Few-Shot Examples**: Including examples in prompts for better results
3. **Structured Output Formatting**: Clear instructions for response formatting

### Monitoring and Analytics

The system tracks performance metrics:

- Response times
- Token usage
- Cache hit rates
- User satisfaction metrics

## 🔮 Future Improvements

### Technical Enhancements

1. **Model Fine-Tuning**
   - Opportunity: Fine-tune models on domain-specific data
   - Benefit: Improved performance for specialized tasks

2. **Advanced RAG Techniques**
   - Opportunity: Implement hybrid retrieval, re-ranking, and query expansion
   - Benefit: More accurate and relevant document retrieval

3. **Multimodal Capabilities Expansion**
   - Opportunity: Enhance image and video processing with specialized models
   - Benefit: More sophisticated analysis of visual content

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 💡 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push the branch (`git push origin feature-branch`).
5. Open a pull request.

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Write unit tests for new features
- Update documentation to reflect changes
- Add type hints to function signatures

## 📞 Contact

For any inquiries, feel free to reach out via GitHub Issues or email at `akshitsharma7093@gmail.com`.

---

🌟 If you like this project, consider giving it a star! ⭐

