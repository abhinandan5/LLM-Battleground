# 🏛️ LLM Arena: Multi-Model Evaluation Pipeline(GPT Vs Gemini)

An advanced, production-grade Retrieval-Augmented Generation (RAG) architecture built to evaluate Large Language Models dynamically. This pipeline ingests multi-format data, applies Cross-Encoder re-ranking, and pits models (like GPT-4o-mini and Gemini Flash) against each other, using Anthropic's Claude as an impartial "LLM-as-a-Judge."

## 🚀 Key Features

*   **Multimodal Data Ingestion:** Dynamically processes PDFs, CSVs, TXT files, YouTube URLs (transcript extraction), and Images (via Gemini Vision OCR/Pre-processing).
*   **Advanced RAG Architecture:** Utilizes `all-MiniLM-L6-v2` for dense vector retrieval via ChromaDB, layered with a `BAAI/bge-reranker-base` Cross-Encoder to guarantee high-fidelity context injection.
*   **Dynamic LLM-as-a-Judge:** Swap the evaluating agent on the fly (Claude 3.5 Sonnet, Haiku, or GPT-4o). The judge parses outputs and returns strict JSON scoring for *Accuracy* and *Completeness*.
*   **Conversational Memory:** Maintains session state for complex, multi-hop follow-up queries.
*   **MLOps Guardrails:** Built-in validation framework to catch model refusals and formatting hallucinations before evaluation.
*   **Audit & Telemetry:** Features a live rolling scoreboard and an exportable CSV audit trail for data compliance.

## 🗂️ Project Architecture

```text
rag-judge-arena/
├── src/
│   ├── engine/
│   │   ├── retriever.py    # ETL Routing, ChromaDB, & Re-ranking pipeline
│   │   └── models.py       # API wrappers with exponential backoff
│   ├── eval/
│   │   ├── judge.py        # LLM-as-a-Judge logic & JSON prompting
│   │   └── validation.py   # MLOps output guardrails
│   ├── ui/
│   │   └── app.py          # Streamlit dashboard & session state management
│   └── utils/
│       ├── helpers.py      # Regex JSON extractors
│       └── exporter.py     # Pandas CSV audit compiler
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/rag-judge-arena.git](https://github.com/yourusername/rag-judge-arena.git)
   cd rag-judge-arena
   
```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_key
   GOOGLE_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   
```

4. **Run the Application:**
   Execute the Streamlit app from the root directory:
   ```bash
   streamlit run src/ui/app.py
   ```
```