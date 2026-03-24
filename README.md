# Multimodal AI Knowledge Assistant

An intelligent AI assistant that processes text, images, and PDFs using 
Agentic AI workflows for multi-step task execution.

## 🚀 Live Demo
👉 [Try the app here](https://your-app-link.streamlit.app)

## Features
- 📄 Document Q&A — Upload PDFs and ask questions
- 🖼️ Image Analysis — Ask questions about images
- 💬 General Chat — Ask anything
- 🔍 Semantic search across 10k+ documents using FAISS
- 📚 Source citations with page numbers
- 💬 Chat history

## Tech Stack
- **LLM:** Groq (LLaMA 3.3 70B)
- **Vision:** Google Gemini
- **Framework:** LangChain
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace (sentence-transformers)
- **UI:** Streamlit

## How It Works
1. Upload PDF documents or images
2. Documents split into chunks and embedded using HuggingFace
3. Chunks stored in FAISS for semantic search
4. Questions matched to relevant chunks
5. LLM generates accurate, context-aware answers

## Setup
```bash
pip install -r requirements.txt
streamlit run multimodal_app.py
```

## Environment Variables
```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```
