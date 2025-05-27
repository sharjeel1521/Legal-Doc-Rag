# 📄 Legal Document RAG System

A Retrieval-Augmented Generation (RAG) system designed to handle complex legal document analysis and Q&A. It combines **semantic and hierarchical chunking**, metadata enrichment, intent classification, and LLM-based generation to return highly contextual answers from legal corpora.

---

## 🧠 Core Components

### 1. Intelligent Chunking (Semantic + Hierarchical Hybrid)

- **Hierarchical Chunking**: Preserves legal structure (e.g., sections, clauses, articles).
- **Semantic Chunking**: Splits based on meaning and coherence using sentence embeddings.
- **Intelligent Chunking**: Hybrid strategy grouping chunks by:
  - Country-specific provisions
  - Legal coherence and clause interrelation

> ✅ Helps preserve legal context and improves retrieval precision.

---

### 2. Metadata Enrichment

- **RAKE**: Extracts key phrases from each chunk.
- **SpaCy**: Identifies named entities (jurisdictions, organizations, dates).
- **Output**: Each chunk is stored as structured JSON:

```json
{
  "chunk_id": "sec_4.1",
  "text": "...",
  "country": "France",
  "entities": ["Apple", "Vodafone"],
  "keywords": ["termination clause", "penalty", "delivery deadline"]
}
# 🔍 Legal Document RAG System – Semantic Retrieval & Intelligent Prompting

This RAG architecture enables accurate, explainable legal question answering with minimal hallucination. It combines semantic retrieval, intelligent chunking, and LLM-powered reasoning with structured metadata filtering.

---

## 3. Semantic Retrieval Engine

- **Embedding Model**: `all-MiniLM-L6-v2` (via SentenceTransformers)
- **Vector Store**: FAISS / Milvus
- **Retrieval Strategy**:
  - **Top-K search** using cosine similarity
  - **Filtering**: Optional metadata filters (e.g., `country = "France"`)

---

## 4. Intent Recognition & Synonym Masking

### **Intent Classifier (LLM-based)**
Detects user query categories such as:
- `Termination`
- `Payment`
- `Obligations`

### **Prompt Masking Engine**
- Rewrites queries using **legal synonyms and phrases**
- Helps reduce embedding mismatch and improve retrieval precision

#### 🔁 Example
- **Original**: "What if the product is late?"
- **Masked**: "What happens if shipment is not on time?"

---

## 5. Prompting & Answer Generation

- Retrieved chunks and metadata are passed to an LLM (e.g., GPT-4)
- A structured prompt includes:
  - User’s (masked) question
  - Retrieved clause texts
  - Intent + Metadata context

### 🧾 Example Prompt Template
