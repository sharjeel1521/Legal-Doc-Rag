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
def classify_question_intent(question, debug=True):
    """Classify question intent using LLM"""
    try:
        # Prepare the classification prompt with emphasis on term patterns
        prompt = f"""
        Analyze this question carefully for term and termination patterns,commercial and financial terms patterns, then classify it:

        Question: "{question}"

        Look specifically for these patterns:
        - Questions about when something ends/terminates
        - Questions about dates of termination
        - Questions about contract duration
        - Questions about renewal or extension
        - Questions about termination conditions
        - Questions about financial terms (pricing, payments, funds)
        - Questions about commercial requirements (order quantities, commitments)
        - Questions about program terms and definitions
        - Questions about country classifications
        - Questions about marketing and advertising obligations
        - Questions about amendments in the term or duration or expiry or termination

        Categories:
        1. TERM - Questions about time periods, dates, duration, renewal
           Sub-categories (if TERM is detected):
           - TERMINATION_DATE (When does it terminate/end?)
           - TERMINATION_CONDITION (How/Why can it be terminated?)
           - RENEWAL (Can it be renewed/extended?)
           - DURATION (How long is the term?)

        2. AMENDMENT - Questions about amendments executed in a particular document

        3. TERM AMENDMENT - Questions about Amendments executed in a document related to Term
           Sub-categories:
           - EXPIRATION_DATE_OF_AGREEMENT (When does it expire/end?)
             Expiry sub-category:
             - Amendment (When was the amendment in the document effective from?)

             To answer questions about amended expiration dates of an agreement:

             Focus on clauses introduced via an Amendment, Addendum, or similar modification.
             Look specifically for changes to the Term section of the original agreement (e.g., "Section 18.1 (Term)").
             Identify if the section contains language such as:
             “Shall be deleted in its entirety and replaced with...”, “Terminate automatically on...”, “Renew automatically unless...”
             Extract the new expiration/termination date mentioned in the amendment and ignore the original clause.

        4. FINANCIAL - Questions about monetary terms, funds, payments
           Sub-categories:
           - AD FUND (Marketing/advertising/carrier fund contributions)
             Carrier sub-categories:
             - AD_FUND WITH AMOUNT (When specific amount/percentage is mentioned/not mentioned explicitly but question is about fund)
               Amount patterns to detect:
               - Euro amounts (e.g., €3,000,000, 3.000.000€, EUR 3000000)
               - Percentages (e.g., 5%, 3.5%)
               - Numbers with currency symbols

           - PRICING (Prices, costs, fees)
           - Carrier (Carrier fund contributions/Authorized Country Funds)
           - MERCHANDISING (Merchandising funds/requirements)

        For AD_FUND questions, carefully check for:
            - Euro symbol (€) followed by numbers
            - Numbers followed by Euro symbol (€)
            - EUR followed by numbers
            - Numbers with dots or commas as thousand separators
            - Percentage symbols (%)

        5. RESTRICTION - Questions about limitations, prohibitions
        6. LIABILITY - Questions about responsibility, damages, warranty
        7. COMMERCIAL - Questions about business requirements
           Sub-categories:
           - ORDER_QUANTITY (Minimum orders, volumes)

        8. PROGRAM - Questions about specific programs or defined terms
           Sub-categories:
           - DEFINITION OF COMPLETE TERM IN THE PROGRAM (Meaning of the program)
           - REQUIREMENTS (Program rules)
        9. Signature - Questions about when specific agreements were signed
           Sub-categories:
           - DATE (Identify the agreement and extract the exact date it was signed from the text)
        10. GENERAL - If none of the above fit

        If the question contains ANY termination or duration patterns, classify it as TERM.

        Respond ONLY with a JSON object:
        {{
            "primary_intent": "CATEGORY",
            "sub_intent": "SUB_CATEGORY",  // Required for TERM category
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation focusing on term patterns if found",
            "keywords": ["relevant", "words", "found"]

        }}
        """

        # Get LLM classification
        response = make_ollama_request(prompt)

        try:
            result = json.loads(response)

            # Validate TERM classification has sub_intent
            if result.get('primary_intent') == 'TERM':
                result['sub_intent'] = 'TERMINATION DATE'  # Default to date if not specified

            if debug:
                print("\nIntent Classification:")
                print(f"Primary Intent: {result.get('primary_intent', 'GENERAL')}")
                print(f"Sub Intent: {result.get('sub_intent', 'NONE')}")
                print(f"Confidence: {result.get('confidence', 0.3):.2f}")
                print(f"Reason: {result.get('reasoning', 'No reason provided')}")
                print(f"Keywords: {result.get('keywords', [])}")

            return {
                "primary_intent": result.get('primary_intent', 'GENERAL'),
                "sub_intent": result.get('sub_intent', 'NONE'),
                "confidence": float(result.get('confidence', 0.3)),
                "matches": {
                    "reason": result.get('reasoning', ''),
                    "keywords": result.get('keywords', [])
                }
            }

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response: {str(e)}")
            return {
                "primary_intent": "GENERAL",
                "sub_intent": "NONE",
                "confidence": 0.3,
                "matches": {
                    "reason": "Error in classification",
                    "keywords": []
                }
            }

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return {
            "primary_intent": "GENERAL",
            "sub_intent": "NONE",
            "confidence": 0.3,
            "matches": {
                "reason": "Error in classification",
                "keywords": []
            }
        }

- Retrieved chunks and metadata are passed to an LLM (e.g., GPT-4)
- A structured prompt includes:
  - User’s (masked) question
  - Retrieved clause texts
  - Intent + Metadata context

### 🧾 Example Prompt Template
