import json
import re
import requests
import numpy as np
import pandas as pd
import csv
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus import utility
import os
from nltk import sent_tokenize
from collections import Counter
import nltk
import certifi
import ssl
import traceback
from rake_nltk import Rake
from typing import AsyncGenerator
import aiohttp
from bs4 import BeautifulSoup
import asyncio
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import PyPDF2
from flask import jsonify
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
# Add required NLTK downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Add at the top of the file
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

file_path = '/Users/taha/Downloads/textchunking/data.xlsx'

# Load the Excel file
xls = pd.ExcelFile(file_path)

# Read the first sheet by index and specify the columns to read
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], usecols=['Questions', 'Specific Document to test if exact answer required'])

# Display the size of the DataFrame
print(f"DataFrame size: {df.shape}")

# Constants and file paths
VECTOR_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
ABBREVIATION_DICT = {
    "CoA": "Contract of Adherence",
    "NDA": "Non-Disclosure Agreement",
    "T&C": "Terms and Conditions",
    "PP": "Price Protection"
}

# Function to expand abbreviations
def expand_abbreviations(text: str) -> str:
    for abbr, full_form in ABBREVIATION_DICT.items():
        text = text.replace(abbr, f"{abbr} ({full_form})")
    return text

# Function for Multi-turn Reference Resolution
def resolve_references(query: str, history: list) -> str:
    if "it" in query or "them" in query:
        for past_query in reversed(history):
            match = re.search(r"\b([A-Za-z]+)\b", past_query)
            if match:
                query = query.replace("it", match.group(1)).replace("them", match.group(1))
                break
    return query



json_file_path = df['Specific Document to test if exact answer required'].iloc[0]

nlp = spacy.load("en_core_web_sm")

def extract_countries(text):

    if not text.strip():
        return set()

    doc = nlp(text)

    return {ent.text for ent in doc.ents if ent.label_ == "GPE"}

RELEVANT_KEYWORDS = {"authorized country", "offer", "pricing", "commitment plan", "obligation"}

def has_relevant_context(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in RELEVANT_KEYWORDS)

# def extract_text_from_json(json_file_path):
#     """Extracts structured and plain text from a given JSON file."""
#     with open(json_file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     output = []

#     for page in data["output"]:
#         page_number = page["pageNum"]
#         plain_text = page.get("plainText", "").strip()

#         contents = [
#             item["content"].strip()
#             for item in page.get("structures", [])
#             if isinstance(item.get("content"), str) and item["content"].strip() != plain_text
#         ]

#         combined_content = " ".join(contents)
#         output.append(f"{combined_content}, {plain_text}")

#     return "\n".join(output)

def extract_text_from_json(json_file_path):

    if isinstance(json_file_path, str):
        paths = [path.strip() for path in re.split(r'[;,]', json_file_path) if path.strip()]
    else:
        paths = list(json_file_path)

    combined_output = []

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for page in data.get("output", []):
                plain_text = page.get("plainText", "").strip()

                contents = [
                    item["content"].strip()
                    for item in page.get("structures", [])
                    if isinstance(item.get("content"), str) and item["content"].strip() != plain_text
                ]

                combined_content = " ".join(contents)
                combined_output.append(f"{combined_content}, {plain_text}")
                #combined_output.append("\n\n--- END OF DOCUMENT ---\n\n")
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return "\n".join(combined_output)



def intelligent_chunking(text, max_length=2000, overlap=200):
    try:
        chunks = []
        chunk_counter = 0
        total_chars = 0
        current_country = None

        if isinstance(text, str):
            text = [text]

        for page_text in text:
            paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]

            for paragraph in paragraphs:
                try:
                    sentences = nltk.sent_tokenize(paragraph)
                except:
                    sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if s.strip()]

                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_length = len(sentence)

                    match = re.search(r"Authorized Country:\s*([A-Za-z\s]+)", sentence)
                    if match:
                        current_country = match.group(1).strip()

                    if current_length + sentence_length > max_length and current_chunk:
                        chunk_text = " ".join(current_chunk)

                        if current_country:
                            chunk_text = f"Authorized Country: {current_country}\n{chunk_text}"

                        chunks.append(chunk_text)
                        total_chars += len(chunk_text)
                        chunk_counter += 1

                        if overlap > 0:
                            current_chunk = current_chunk[-1:]
                            current_length = len(current_chunk[0])
                        else:
                            current_chunk = []
                            current_length = 0

                    current_chunk.append(sentence)
                    current_length += sentence_length + 1

                if current_chunk:
                    chunk_text = " ".join(current_chunk)

                    if current_country:
                        chunk_text = f"Authorized Country: {current_country}\n{chunk_text}"

                    chunks.append(chunk_text)
                    total_chars += len(chunk_text)
                    chunk_counter += 1

        hierarchy = []
        chunk_mapping = {}
        current_parent = None
        attach_next_as_child = False

        for chunk_id, chunk_text in enumerate(chunks, start=1):
            is_parent = chunk_text.strip().endswith(":")

            if is_parent:
                current_parent = chunk_id
                chunk_mapping[current_parent] = {"id": chunk_id, "text": chunk_text, "children": []}
                hierarchy.append(chunk_mapping[current_parent])
                attach_next_as_child = True

            elif attach_next_as_child:
                combined_text = chunk_mapping[current_parent]["text"] + "\n" + chunk_text
                chunk_mapping[current_parent]["text"] = combined_text
                chunks[current_parent - 1] = combined_text
                attach_next_as_child = False

            else:
                chunk_mapping[chunk_id] = {"id": chunk_id, "text": chunk_text, "children": []}

        return chunks

    except Exception as e:
        return []

# Initialize Milvus Client and create schema
client = MilvusClient("milvus_demo03.db")  # Updated DB name
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="upload_date", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="entities", dtype=DataType.JSON)
]
schema = CollectionSchema(fields=fields, description="Legal document chunks")

# Check if collection exists and handle accordingly
try:
    if client.has_collection("demo_collection"):
        print("Collection already exists. Dropping existing collection...")
        client.drop_collection("demo_collection")
        print("Creating new collection with updated schema...")

    client.create_collection(
        collection_name="demo_collection",
        schema=schema,
        dimension=384
    )

    # Create index with correct parameter format
    client.create_index(
        collection_name="demo_collection",
        index_params=[{
            "field_name": "embedding",
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }]
    )
    print("Collection and index created successfully")

except Exception as e:
    print(f"Error handling collection: {str(e)}")
    raise

# Function to check the schema of a collection
def check_schema(collection_name):
    try:
        collection_info = utility.get_collection_info(collection_name)
        if 'fields' not in collection_info:
            print(f"No fields found in collection info for {collection_name}")
            return None

        # Create schema dictionary from fields
        schema = {
            field['name']: {
                'type': field['type'],
                'params': field['params']
            }
            for field in collection_info['fields']
        }
        return schema
    except Exception as e:
        print(f"Error retrieving schema: {str(e)}")
        return None

# Function to extract entities from a chunk
def extract_entities_from_chunk(chunk_text):
    """Extract entities from a single chunk"""
    return {
        'dates': re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}', chunk_text),
        'amounts': re.findall(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars)', chunk_text),
        'organizations': re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.|LLC|Ltd\.|Corporation|Corp\.|Company|Co\.))', chunk_text),
        'legal_refs': re.findall(r'(?:Section|Article|Clause)\s+\d+(?:\.\d+)*', chunk_text),
        'defined_terms': re.findall(r'"([^"]+)"\s+(?:shall\s+)?means?', chunk_text)
    }

def extract_chunk_metadata(chunk_text):
    """Extract comprehensive metadata from chunk including legal clauses and key phrases"""
    try:
        # Legal clause categories based on prompt categories
        legal_categories = {
            'term_clauses': [
                r'term\s+of\s+agreement', r'duration', r'period\s+of\s+performance',
                r'effective\s+date', r'termination\s+date', r'expiration'
            ],
            'obligation_clauses': [
                r'shall', r'must', r'required\s+to', r'obligations?',
                r'responsibilities', r'duties', r'undertakes?\s+to'
            ],
            'restriction_clauses': [
                r'shall\s+not', r'prohibited', r'restricted', r'limitation',
                r'may\s+not', r'forbidden', r'unless', r'except'
            ],
            'modification_clauses': [
                r'amendment', r'modification', r'change', r'alter',
                r'revise', r'update', r'written\s+consent'
            ],
            'termination_clauses': [
                r'termination', r'terminate', r'cancellation', r'end',
                r'notice\s+period', r'cause', r'breach'
            ],
            'liability_clauses': [
                r'liability', r'indemnification', r'warranty', r'damages',
                r'compensation', r'responsible\s+for', r'hold\s+harmless'
            ],
            'dispute_resolution': [
                r'dispute', r'arbitration', r'mediation', r'jurisdiction',
                r'governing\s+law', r'venue', r'resolution'
            ]
        }

        # Extract section numbers and headers
        section_pattern = r'(?:Section|Article|Clause)\s+(\d+(?:\.\d+)*)\s*[.-]\s*([^\n]+)'
        sections = re.finditer(section_pattern, chunk_text)
        section_info = [{
            'number': match.group(1),
            'title': match.group(2).strip(),
            'text': chunk_text[match.start():match.end()]
        } for match in sections]

        # Safer NLTK processing
        try:
            tokens = nltk.word_tokenize(chunk_text)
            pos_tags = nltk.pos_tag(tokens)

            # Extract named entities using NLTK
            named_entities = []
            try:
                for chunk in nltk.ne_chunk(pos_tags):
                    if hasattr(chunk, 'label'):
                        named_entities.append({
                            'text': ' '.join(c[0] for c in chunk),
                            'type': chunk.label()
                        })
            except Exception as ne_error:
                print(f"Warning: Named Entity extraction failed: {str(ne_error)}")
                named_entities = []
        except Exception as nltk_error:
            print(f"Warning: NLTK processing failed: {str(nltk_error)}")
            tokens = chunk_text.split()
            pos_tags = []
            named_entities = []

        # RAKE for key phrase extraction
        r = Rake(
            stopwords=set(nltk.corpus.stopwords.words('english')),
            min_length=1,
            max_length=4
        )
        r.extract_keywords_from_text(chunk_text)
        rake_phrases = r.get_ranked_phrases()[:10]  # Top 10 key phrases

        # Legal clause analysis
        legal_matches = {}
        for category, patterns in legal_categories.items():
            matches = []
            for pattern in patterns:
                found = re.finditer(pattern, chunk_text, re.IGNORECASE)
                for match in found:
                    # Get surrounding context (100 chars before and after)
                    start = max(0, match.start() - 100)
                    end = min(len(chunk_text), match.end() + 100)
                    matches.append({
                        'pattern': pattern,
                        'text': chunk_text[start:end].strip(),
                        'position': match.start()
                    })
            if matches:
                legal_matches[category] = matches

        return {
            'chunk_level': {
                'length': len(chunk_text),
                'sentence_count': len(nltk.sent_tokenize(chunk_text)),
                'sections': section_info,
                'key_phrases': {
                    'rake_phrases': rake_phrases,
                    'legal_terms': extract_legal_terms(chunk_text),
                    'custom_phrases': extract_key_phrases(chunk_text)
                },
                'entities': {
                    'dates': extract_dates(chunk_text),
                    'parties': extract_entities_from_chunk(chunk_text).get('organizations', []),
                    'monetary_values': extract_entities_from_chunk(chunk_text).get('amounts', []),
                    'legal_refs': re.findall(r'(?:Section|Article|Clause)\s+\d+(?:\.\d+)*', chunk_text),
                    'named_entities': named_entities
                },
                'legal_clauses': legal_matches,
                'statistics': {
                    'word_count': len(tokens),
                    'unique_terms': len(set(word.lower() for word in tokens)),
                    'legal_terms_count': sum(len(matches) for matches in legal_matches.values())
                }
            }
        }
    except Exception as e:
        print(f"Error extracting chunk metadata: {str(e)}")
        return {'chunk_level': {}}

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def extract_key_phrases(text):
    """Extract key phrases using spaCy, focusing on NER, nouns, and verbs while excluding pronouns."""
    try:
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")

        # Process the text
        doc = nlp(text)

        phrases = []

        # Extract noun phrases (Skip single-character words)
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 1 and len(chunk.text.split()) <= 4:
                phrases.append({
                    'text': chunk.text.strip(),
                    'type': 'noun_phrase'
                })

        # Extract named entities (Skip single-character entities)
        for ent in doc.ents:
            if len(ent.text) > 1:
                phrases.append({
                    'text': ent.text.strip(),
                    'type': f'NER_{ent.label_}'  # Add entity type
                })

        # Extract verb phrases (Avoid single-character words and pronouns)
        for token in doc:
            if token.pos_ == "VERB":
                subj = next((child.text for child in token.children if child.dep_ == "nsubj" and child.pos_ != "PRON"), "")
                obj = next((child.text for child in token.children if child.dep_ in ["dobj", "pobj"]), "")

                if subj and obj:
                    phrase = f"{subj} {token.text} {obj}"
                elif subj:
                    phrase = f"{subj} {token.text}"
                elif obj:
                    phrase = f"{token.text} {obj}"
                else:
                    phrase = token.text

                if len(phrase) > 1 and len(phrase.split()) <= 4:
                    phrases.append({
                        'text': phrase.strip(),
                        'type': 'verb_phrase'
                    })

        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            if phrase['text'].lower() not in seen:
                seen.add(phrase['text'].lower())
                unique_phrases.append(phrase)

        #print(f"\nExtracted {len(unique_phrases)} key phrases:")
       # for phrase in unique_phrases[:5]:  # Show first 5 phrases
        #    print(f"- {phrase['type']}: {phrase['text']}")

        return unique_phrases

    except Exception as e:
        print(f"Error in key phrase extraction: {str(e)}")
        traceback.print_exc()
        return []

def extract_legal_terms(text):
    """Extract common legal terms and definitions"""
    legal_terms = []

    # Common legal term patterns
    patterns = [
        r'"([^"]+)"\s+(?:shall\s+)?means?',  # Defined terms
        r'(?:herein|hereof|hereto|hereunder|hereby)',  # Legal references
        r'(?:aforementioned|aforesaid)',  # References
        r'(?:mutatis\s+mutandis|inter\s+alia|de\s+facto|de\s+jure)',  # Latin terms
        r'(?:force\s+majeure|time\s+is\s+of\s+the\s+essence|without\s+prejudice)'  # Legal concepts
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            legal_terms.append(match.group().strip())

    return list(set(legal_terms))

def extract_dates(text):
    """Extract dates from text with enhanced pattern matching"""
    try:
        date_patterns = {
            'standard': [
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
                r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',    # YYYY/MM/DD
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
                r'Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
            ],
            'relative': [
                r'\b\d+\s+(?:day|month|year)s?\s+(?:from|after|before|prior\s+to)\b',
                r'\b(?:within|after)\s+\d+\s+(?:day|month|year)s?\b'
            ],
            'effective': [
                r'effective\s+(?:date\s+)?(?:of\s+)?([^,.]+)',
                r'commenc(?:es?|ing)\s+(?:on\s+)?([^,.]+)',
                r'begin(?:s|ning)\s+(?:on\s+)?([^,.]+)'
            ],
            'termination': [
                r'terminat(?:es?|ion)\s+(?:date\s+)?(?:of\s+)?([^,.]+)',
                r'expir(?:es?|ation)\s+(?:date\s+)?(?:of\s+)?([^,.]+)',
                r'end(?:s|ing)\s+(?:on\s+)?([^,.]+)'
            ]
        }

        dates = {
            'standard_dates': [],
            'relative_dates': [],
            'effective_dates': [],
            'termination_dates': []
        }

        # Extract dates with context
        for category, patterns in date_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)

                    date_info = {
                        'date': match.group(),
                        'context': text[start:end].strip(),
                        'position': match.start()
                    }

                    if category == 'standard':
                        dates['standard_dates'].append(date_info)
                    elif category == 'relative':
                        dates['relative_dates'].append(date_info)
                    elif category == 'effective':
                        dates['effective_dates'].append(date_info)
                    elif category == 'termination':
                        dates['termination_dates'].append(date_info)

        return dates

    except Exception as e:
        print(f"Error extracting dates: {str(e)}")
        return {
            'standard_dates': [],
            'relative_dates': [],
            'effective_dates': [],
            'termination_dates': []
        }

import ast

def insert_text_with_schema(chunk_id, chunk_text, pdf_path):
    """Insert text with both document and chunk level metadata"""
    try:
        # Get embeddings
        embedding = VECTOR_MODEL.encode([chunk_text])[0]

        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract chunk-specific metadata
        chunk_metadata = extract_chunk_metadata(chunk_text)

        # Combine metadata
        combined_metadata = {
            'document_level': {
                'file_name': os.path.basename(pdf_path),
                'full_path': pdf_path,
                'upload_date': current_date
            },
            **chunk_metadata  # This adds the chunk_level metadata
        }

        # Prepare data with correct field names
        data = {
            'id': chunk_id,
            'embedding': embedding.tolist(),
            'text': chunk_text,
            'file_name': pdf_path,
            'upload_date': current_date,
            'entities': json.dumps(combined_metadata)
        }

        # Insert data
        try:
            client.insert(
                collection_name="demo_collection",
                data=[data]
            )
            print(f"Successfully inserted chunk {chunk_id} with metadata")
            return True
        except Exception as insert_error:
            print(f"Failed to insert chunk {chunk_id}: {str(insert_error)}")
            return False

    except Exception as e:
        print(f"Error preparing data for chunk {chunk_id}: {str(e)}")
        return False

# Test the function with some sample data
insert_text_with_schema(1, "Sample text for insertion into Milvus.", "sample.pdf")

# Modify retrieve_relevant_chunks to handle entity-specific searches
import json
import traceback
from rank_bm25 import BM25Okapi
import ast
#-------------------------------retreive chunks by intent ------------------------------------

#---------------------------------------------------------------------------------------------
def retrieve_relevant_chunks_by_intent(question, intent_info, client, VECTOR_MODEL, top_k=10):
    """Retrieve chunks based on intent category and metadata matching."""
    try:
        # Print input details
        print("\n" + "="*50)
        print("INPUT DETAILS:")
        print("="*50)
        print(f"Original Question: {question}")
        print(f"Intent Info: {intent_info}")

        # Validate and concatenate question with intents
        enhanced_question = question
       # if intent_info and isinstance(intent_info, dict):
       #     primary_intent = intent_info.get('primary_intent', '')
        #    secondary_intent = intent_info.get('sub_intent', '')

         #   if primary_intent:
          #      enhanced_question += f" {primary_intent}"
          #  if secondary_intent:
          #      enhanced_question += f" {secondary_intent}"

        print("\n" + "="*50)
        print("ENHANCED QUERY:")
        print("="*50)
        print(f"Primary Intent: {intent_info.get('primary_intent', 'None')}")
        print(f"Secondary Intent: {intent_info.get('sub_intent', 'None')}")
        print(f"Enhanced Question: {enhanced_question}")

        query_embedding = VECTOR_MODEL.encode([enhanced_question])[0]
        collection_name = "demo_collection"

        if not client.has_collection(collection_name):
            print("Collection does not exist")
            return []

        try:
            client.load_collection(collection_name)
        except Exception as e:
            print(f"Error loading collection: {str(e)}")
            return []

        print("\n" + "="*50)
        print("SEARCHING COLLECTION:")
        print("="*50)

        # First try direct query for termination-related content
        if intent_info.get("primary_intent") == "TERM":
            try:
                results = client.query(
                    collection_name=collection_name,
                    output_fields=["text"],
                    expr="text like '%termination%' OR text like '%term%' OR text like '%expiry%' OR text like '%end date%'",
                    consistency_level="Strong"
                )

                if results:
                    print(f"Found {len(results)} direct matches")
                    processed_results = []
                    for hit in results:
                        text = hit.get("text", "")
                        if text:
                            processed_results.append({
                                "text": text,
                                "score": 1.0
                            })
                    if processed_results:
                        return processed_results[:top_k]
            except Exception as e:
                print(f"Direct query failed: {str(e)}")

        # Fallback to vector search
        try:
            search_results = client.search(
                collection_name=collection_name,
                data=[query_embedding.tolist()],
                anns_field="embedding",
                params={
                    "metric_type": "COSINE",
                    "nprobe": 10
                },
                limit=2,
                output_fields=["text", "file_name"]
            )
            print("\n" + "="*50)
            print("SEARCHING chunks:")
            print("="*50)

            print(search_results)
            if not search_results or not isinstance(search_results, list) or len(search_results) == 0:
                print("No search results found")
                return []

            results_list = search_results[0]
            if not results_list:
                print("Empty results list")
                return []

            processed_results = []
            termination_keywords = ["term", "terminate", "termination", "end", "expiry", "expiration", "duration", "period"]
            has_termination = intent_info.get("primary_intent") == "TERM"

            for hit in results_list:
                try:
                    text = hit.get("entity", {}).get("text", "")
                    if not text:
                        continue

                    distance = float(hit.get("distance", 1.0))
                    score = 1.0 - distance

                    # Boost score for termination-related content
                    if has_termination:
                        has_term = any(keyword in text.lower() for keyword in termination_keywords)
                        if has_term:
                            score += 0.5
                            print(f"Found termination-related content with score {score}")

                    processed_results.append({
                        "text": text,
                        "score": min(score, 1.0),
                        "file_name": file_name
                    })

                except Exception as e:
                    print(f"Error processing hit: {str(e)}")
                    continue

            # Sort by score and filter if needed
            processed_results = sorted(processed_results, key=lambda x: x["score"], reverse=True)

            if has_termination:
                filtered_results = [r for r in processed_results
                                  if any(keyword in r["text"].lower() for keyword in termination_keywords)]
                if filtered_results:
                    print(f"\nReturning {len(filtered_results[:top_k])} filtered results for termination-related content")
                    return filtered_results[:top_k]

            print(f"\nReturning {len(processed_results[:top_k])} results from vector search")
            return processed_results[:top_k]

        except Exception as e:
            print(f"Vector search failed: {str(e)}")
            return []

    except Exception as e:
        print(f"Error in retrieve_relevant_chunks_by_intent: {str(e)}")
        traceback.print_exc()
        return []

def calculate_metadata_score(entities, text, intent_info):
    """Calculate metadata match score based on intent with improved scoring"""
    try:
        score = 0.0
        print("\n" + "-"*40)
        print("METADATA SCORING DEBUG")
        print("-"*40)
        print(f"Intent: {intent_info['primary_intent']}")
        print(f"Initial Score: {score}")

        # Debug entities
        print("\nEntities received:")
        print(json.dumps(entities, indent=2))

        if intent_info["primary_intent"] == "TERM":
            print("\nChecking TERM intent specific metadata:")

            # Check for dates
            if entities.get('dates'):
                print("✓ Found dates:", entities['dates'])
                score += 0.2
                print(f"Score after dates: {score}")

            # Check for termination date
            if entities.get('termination_date'):
                print("✓ Found termination date:", entities['termination_date'])
                score += 0.3
                print(f"Score after termination date: {score}")

            # Check for term-related keywords in text
            term_keywords = ["term", "duration", "period", "expire", "termination"]
            found_keywords = [kw for kw in term_keywords if kw in text.lower()]
            if found_keywords:
                print("✓ Found term keywords:", found_keywords)
                score += 0.2
                print(f"Score after keywords: {score}")

            # Check for document type
            if entities.get('doc_type') in ['AGREEMENT', 'CONTRACT']:
                print("✓ Found relevant document type:", entities.get('doc_type'))
                score += 0.1
                print(f"Score after doc type: {score}")

        elif intent_info["primary_intent"] == "OBLIGATION":
            print("\nChecking OBLIGATION intent specific metadata:")

            obligation_terms = ["shall", "must", "required", "obligation"]
            found_terms = [term for term in obligation_terms if term in text.lower()]
            if found_terms:
                print("✓ Found obligation terms:", found_terms)
                score += 0.4
                print(f"Score after obligation terms: {score}")

            if entities.get('obligations'):
                print("✓ Found obligation metadata:", entities['obligations'])
                score += 0.3
                print(f"Score after obligation metadata: {score}")

        elif intent_info["primary_intent"] == "RESTRICTION":
            print("\nChecking RESTRICTION intent specific metadata:")

            restriction_terms = ["not", "prohibited", "restricted", "limitation"]
            found_terms = [term for term in restriction_terms if term in text.lower()]
            if found_terms:
                print("✓ Found restriction terms:", found_terms)
                score += 0.4
                print(f"Score after restriction terms: {score}")

            if entities.get('restrictions'):
                print("✓ Found restriction metadata:", entities['restrictions'])
                score += 0.3
                print(f"Score after restriction metadata: {score}")

        elif intent_info["primary_intent"] == "LIABILITY":
            print("\nChecking LIABILITY intent specific metadata:")

            liability_terms = ["liable", "liability", "warranty", "damages"]
            found_terms = [term for term in liability_terms if term in text.lower()]
            if found_terms:
                print("✓ Found liability terms:", found_terms)
                score += 0.4
                print(f"Score after liability terms: {score}")

            if entities.get('liability'):
                print("✓ Found liability metadata:", entities['liability'])
                score += 0.3
                print(f"Score after liability metadata: {score}")

        # Ensure score doesn't exceed 1.0
        final_score = min(score, 1.0)

        print("\nScoring Summary:")
        print("-"*40)
        print(f"Raw Score: {score}")
        print(f"Final Score (capped): {final_score}")
        print("-"*40)

        return final_score

    except Exception as e:
        print("\n❌ Error in metadata scoring:")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("Returning default score of 0.0")
        return 0.0

# def make_jamba_request(prompt, app_id="199323", app_password="xeuj2raw7nxsfekf", temperature=0.3, top_p=0.95, max_tokens=4096):
#     """Make request to Jamba 1.6 API"""
#     try:
#         # First API call to get authentication token
#         url = "https://idmsservice.corp.apple.com/apptoapp/token/generate"
#         payload = json.dumps({
#             "appId": app_id,
#             "appPassword": app_password,
#             "context": "shuri-proxy",
#             "otherApp": 152064,
#             "oneTimeToken": False
#         })
#         headers = {
#             "Content-Type": "application/json",
#             "Cookie": "site=USA; vet=17A53AD00FD645735AC5F8B7A5EE9DE5; dslang=US-EN; site=USA; vet=2AF800C43A8A11AD9093209604D86529"
#         }
#         response = requests.post(url, headers=headers, data=payload)

#         # Check for a valid response
#         if response.status_code != 200:
#             print("Error: Failed to retrieve token")
#             return None

#         token = response.json().get("token")
#         if not token:
#             print("Error: No token received")
#             return None

#         # Second API call to Jamba 1.6 endpoint
#         url = "https://llm-gateway-aws02w.shuri.apple.com/shuri/evt/jamba16largevllm/v01/chat/completions"
#         payload = json.dumps({
#             "model": "AI21-Jamba-1.6-Large",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             "temperature": temperature,
#             "top_p": top_p,
#             "max_tokens": max_tokens,
#             "stream": False
#         })
#         headers = {
#             "appId": str(app_id),
#             "token": token,
#             "Content-Type": "application/json",
#             "Cookie": "site=USA; dslang=US-EN"
#         }
#         response = requests.post(url, headers=headers, data=payload)

#         if response.status_code != 200:
#             print("Error: Failed to retrieve response")
#             return None

#         # Extract and clean the output
#         try:
#             response_data = response.json()
#             output = response_data["choices"][0]["message"]["content"]

#             # Clean the output to ensure it's valid JSON
#             output = output.strip()

#             # Remove any markdown code block markers if present
#             output = output.replace('```json', '').replace('```', '').strip()

#             # Remove any text before the first { and after the last }
#             if '{' in output and '}' in output:
#                 output = output[output.find('{'):output.rfind('}')+1]

#             # Replace any smart quotes with regular quotes
#             output = output.replace('"', '"').replace('"', '"')

#             # Remove any trailing commas before closing braces
#             output = re.sub(r',(\s*[}\]])', r'\1', output)

#             # Validate that the output is valid JSON
#             try:
#                 json.loads(output)
#                 return output
#             except json.JSONDecodeError as e:
#                 print(f"Error validating JSON output: {str(e)}")
#                 print(f"Raw output: {output}")
#                 return None

#         except (KeyError, IndexError) as e:
#             print(f"Error parsing response: {str(e)}")
#             print(f"Response data: {response_data}")
#             return None

#     except Exception as e:
#         print(f"Error in Jamba request: {str(e)}")
#         return None


def make_ollama_request(prompt):
    """Make request to Ollama API"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'qwen2.5:32b',
                                   'prompt': prompt,
                                   'stream': False
                               })
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        print(f"Error in Ollama request: {str(e)}")
        return None

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def extract_entities_from_question(question):
    """Extract named entities from the question"""
    try:
        doc = nlp(question)
        entities = {
            'ORG': [],
            'DATE': [],
            'MONEY': [],
            'PERSON': [],
            'GPE': [],  # Geographical/Political Entities
            'LAW': [],  # Legal references
            'CARDINAL': []  # Numbers
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        return entities
    except Exception as e:
        print(f"Error extracting entities from question: {str(e)}")
        return {}

def generate_synonyms(word):
    """Generate synonyms for a given word using NLTK WordNet"""
    try:
        from nltk.corpus import wordnet
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        return list(set(synonyms))
    except Exception as e:
        print(f"Error generating synonyms: {str(e)}")
        return []

def match_question_phrases(question, text):
    """Match question phrases with text and return matching phrases"""
    try:
        # Tokenize question into phrases
        question_tokens = nltk.word_tokenize(question.lower())
        text_tokens = nltk.word_tokenize(text.lower())

        # Generate phrases of different lengths
        question_phrases = []
        for i in range(1, 4):  # Generate phrases of length 1-3
            for j in range(len(question_tokens) - i + 1):
                phrase = ' '.join(question_tokens[j:j+i])
                question_phrases.append(phrase)

        # Find matching phrases in text
        matching_phrases = []
        for phrase in question_phrases:
            if phrase in text.lower():
                matching_phrases.append(phrase)

        return matching_phrases
    except Exception as e:
        print(f"Error matching phrases: {str(e)}")
        return []

def llm_judge_synonyms_and_phrases(question, synonyms, matching_phrases, context):
    """Use LLM to judge the relevance of synonyms and matching phrases"""
    try:
        # Prepare prompt for LLM with explicit JSON formatting instructions
        prompt = f"""
        Analyze the following question, synonyms, and matching phrases to determine their relevance and appropriateness.
        Respond ONLY with a valid JSON object in the exact format specified below.

        Question: "{question}"

        Synonyms:
        {json.dumps(synonyms, indent=2)}

        Matching Phrases:
        {json.dumps(matching_phrases, indent=2)}

        Context Excerpt:
        {context[:500]}...  # First 500 chars of context

        Evaluate each synonym and matching phrase for:
        1. Semantic relevance to the question
        2. Contextual appropriateness
        3. Potential impact on question understanding

        You MUST respond with a JSON object in this exact format:
        {{
            "relevant_synonyms": {{
                "word": ["approved", "synonyms"]
            }},
            "relevant_phrases": ["approved", "phrases"],
            "reasoning": "Brief explanation of the evaluation"
        }}

        Do not include any text before or after the JSON object.
        Ensure all quotes are straight quotes (") not curly quotes (").
        """

        # Get LLM evaluation
        response = make_ollama_request(prompt)

        if not response:
            print("No response received from LLM")
            return {
                "relevant_synonyms": synonyms,
                "relevant_phrases": matching_phrases,
                "reasoning": "No response from LLM"
            }

        try:
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            # Remove any markdown code block markers if present
            response = response.replace('```json', '').replace('```', '').strip()
            # Remove any text before the first {
            response = response[response.find('{'):]
            # Remove any text after the last }
            response = response[:response.rfind('}')+1]

            evaluation = json.loads(response)

            # Validate the response structure
            if not isinstance(evaluation, dict):
                raise ValueError("Response is not a dictionary")

            if "relevant_synonyms" not in evaluation:
                evaluation["relevant_synonyms"] = synonyms
            if "relevant_phrases" not in evaluation:
                evaluation["relevant_phrases"] = matching_phrases
            if "reasoning" not in evaluation:
                evaluation["reasoning"] = "No reasoning provided"

            return evaluation

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "relevant_synonyms": synonyms,
                "relevant_phrases": matching_phrases,
                "reasoning": f"Failed to parse LLM response: {str(e)}"
            }

    except Exception as e:
        print(f"Error in LLM judging: {str(e)}")
        traceback.print_exc()
        return {
            "relevant_synonyms": synonyms,
            "relevant_phrases": matching_phrases,
            "reasoning": f"Error in evaluation: {str(e)}"
        }

def enhance_question_with_llm(question, context):
    """Enhance question using LLM to make it more specific and comprehensive"""
    try:
        prompt = f"""
        Analyze and enhance this question to make it more specific and comprehensive.
        Consider the context and add relevant details that would help in finding a more accurate answer.

        Original Question: "{question}"

        Context Excerpt:
        {context[:500]}...  # First 500 chars of context

        Enhance the question by:
        1. Adding relevant context from the document
        2. Making it more specific and clear
        3. Including important related terms
        4. Maintaining the original intent

        Respond ONLY with a JSON object in this format:
        {{
            "enhanced_question": "The enhanced version of the question",
            "added_context": ["List of context elements added"],
            "key_terms": ["Important terms included"],
            "reasoning": "Brief explanation of the enhancements"
        }}

        Do not include any text before or after the JSON object.
        Ensure all quotes are straight quotes (") not curly quotes (").
        """

        response = make_ollama_request(prompt)

        if not response:
            print("No response received from LLM for question enhancement")
            return {
                "enhanced_question": question,
                "added_context": [],
                "key_terms": [],
                "reasoning": "No enhancement performed"
            }

        try:
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            response = response.replace('```json', '').replace('```', '').strip()
            response = response[response.find('{'):]
            response = response[:response.rfind('}')+1]

            enhancement = json.loads(response)

            # Validate the response structure
            if not isinstance(enhancement, dict):
                raise ValueError("Response is not a dictionary")

            if "enhanced_question" not in enhancement:
                enhancement["enhanced_question"] = question
            if "added_context" not in enhancement:
                enhancement["added_context"] = []
            if "key_terms" not in enhancement:
                enhancement["key_terms"] = []
            if "reasoning" not in enhancement:
                enhancement["reasoning"] = "No reasoning provided"

            return enhancement

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM enhancement response: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "enhanced_question": question,
                "added_context": [],
                "key_terms": [],
                "reasoning": f"Failed to parse LLM response: {str(e)}"
            }

    except Exception as e:
        print(f"Error in question enhancement: {str(e)}")
        traceback.print_exc()
        return {
            "enhanced_question": question,
            "added_context": [],
            "key_terms": [],
            "reasoning": f"Error in enhancement: {str(e)}"
        }

def process_question_with_llm_intent(question, context, client, VECTOR_MODEL, debug=True):
    """Process question with enhanced NER and intent-based retrieval"""
    try:
        # First enhance the question using LLM
        enhanced_question_info = enhance_question_with_llm(question, context)
        enhanced_question = enhanced_question_info["enhanced_question"]

        # Extract entities from enhanced question
        question_entities = extract_entities_from_question(enhanced_question)

        # Generate synonyms for key words in enhanced question
        question_words = nltk.word_tokenize(enhanced_question)
        synonyms = {}
        for word in question_words:
            if len(word) > 3:  # Only generate synonyms for words longer than 3 characters
                word_synonyms = generate_synonyms(word)
                if word_synonyms:
                    synonyms[word] = word_synonyms

        # Match phrases in enhanced question with context
        matching_phrases = match_question_phrases(enhanced_question, context)

        # Get LLM evaluation of synonyms and phrases
        llm_evaluation = llm_judge_synonyms_and_phrases(enhanced_question, synonyms, matching_phrases, context)

        # Get intent classification using enhanced question
        intent_info = classify_question_intent(enhanced_question)

        # Print input details
        print("\n" + "="*50)
        print("INPUT DETAILS:")
        print("="*50)
        print(f"Original Question: {question}")
        print(f"Enhanced Question: {enhanced_question}")
        print(f"Added Context: {enhanced_question_info['added_context']}")
        print(f"Key Terms: {enhanced_question_info['key_terms']}")
        print(f"Enhancement Reasoning: {enhanced_question_info['reasoning']}")
        print(f"Intent Info: {intent_info}")

        # Print LLM evaluation results
        print("\nLLM Evaluation Results:")
        print("-"*40)
        print("Relevant Synonyms:")
        for word, approved_synonyms in llm_evaluation.get("relevant_synonyms", {}).items():
            print(f"{word}: {', '.join(approved_synonyms)}")

        print("\nRelevant Phrases:")
        for phrase in llm_evaluation.get("relevant_phrases", []):
            print(f"- {phrase}")

        print("\nEvaluation Reasoning:")
        print(llm_evaluation.get("reasoning", "No reasoning provided"))

        # Further enhance the question with intents and approved synonyms/phrases
        final_enhanced_question = enhanced_question
        if intent_info and isinstance(intent_info, dict):
            primary_intent = intent_info.get('primary_intent', '')
            secondary_intent = intent_info.get('sub_intent', '')

            # Add intents
            if primary_intent:
                final_enhanced_question += f" {primary_intent}"
            if secondary_intent:
                final_enhanced_question += f" {secondary_intent}"

            # Add approved matching phrases
            if llm_evaluation.get("relevant_phrases"):
                final_enhanced_question += f" {' '.join(llm_evaluation['relevant_phrases'])}"

            # Add approved synonyms for key words
            for word, approved_synonyms in llm_evaluation.get("relevant_synonyms", {}).items():
                if approved_synonyms:
                    final_enhanced_question += f" {' '.join(approved_synonyms)}"

        print("\n" + "="*50)
        print("FINAL ENHANCED QUERY:")
        print("="*50)
        print(f"Primary Intent: {intent_info.get('primary_intent', 'None')}")
        print(f"Secondary Intent: {intent_info.get('sub_intent', 'None')}")
        print(f"Final Enhanced Question: {final_enhanced_question}")

        # Get relevant chunks using the final enhanced question
        chunks = retrieve_relevant_chunks_by_intent(final_enhanced_question, intent_info, client, VECTOR_MODEL)

        # Calculate confidence scores
        confidence_info = calculate_confidence(intent_info, chunks)

        # Generate answer using the chunks
          # Enhanced prompt for LLM

        # Enhanced prompt for LLM
        prompt = f"""
        Question: {final_enhanced_question}

        Document Context:
        {chunks}

        Task: Based on the provided context, answer the question. Consider:
        1. Relevance of each chunk (indicated by score)
        2. Key phrases in each chunk
        3. Document context and relationships
        4. Specific details and dates if present

        If multiple possible answers exist, explain why you chose a particular one.
        If the answer isn't clear from the context, say so.

        Format your response as JSON with these keys:
        {{
            "answer": "your detailed answer",
            "confidence": "high/medium/low",
            "reasoning": "explanation of your answer",
            "source_chunks": ["chunk numbers used"],
            "key_phrases_used": ["relevant key phrases that led to the answer"]
        }}
        """

        # Get LLM response
        answer_text = make_ollama_request(prompt)


        #answer_text = " ".join(chunk.get('text', '') for chunk in chunks[:3]) if chunks else "No relevant information found."
        #answer_text = generate_answer_from_chunks(chunks, final_enhanced_question)
        # Return structured response
        return {
            'intent': intent_info,
            'confidence': confidence_info,
            'chunks': chunks,
            'answer': answer_text,
            'enhanced_question': final_enhanced_question,
            'original_question': question,
            'enhancement_info': enhanced_question_info,
            'entities': question_entities,
            'synonyms': synonyms,
            'matching_phrases': matching_phrases
        }

    except Exception as e:
        print(f"Error in question processing: {str(e)}")
        traceback.print_exc()
        # Return default response instead of None
        return {
            'intent': {'primary_intent': 'GENERAL'},
            'confidence': {'overall': 0.3},
            'chunks': [],
            'answer': f"Error processing question: {str(e)}",
            'enhanced_question': question,
            'original_question': question,
            'enhancement_info': {
                'enhanced_question': question,
                'added_context': [],
                'key_terms': [],
                'reasoning': "Error in enhancement"
            },
            'entities': {},
            'synonyms': {},
            'matching_phrases': []
        }

def calculate_confidence(intent_info, chunks):
    """Calculate overall confidence score"""
    intent_confidence = float(intent_info.get('confidence', 0))
    chunk_confidence = max(chunk['score'] for chunk in chunks) if chunks else 0

    return {
        "overall": (intent_confidence + chunk_confidence) / 2,
        "intent": intent_confidence,
        "retrieval": chunk_confidence
    }

#-------------Classify question intent using lLM--------------------------------------------
#-------------------------------------------------------------------------------------------
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
#--------------Main--------------------------------------------------------------------------------
# Update the main processing loop
#---------------------------------------------------------------------------------------------------
output_file = 'output.csv'
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Document', 'Question', 'Intent', 'Chunks Used', 'Response'])

    for index, row in df.head(1).iterrows():
        try:
            print("\n" + "="*80)
            print(f"DOCUMENT {index + 1}")
            print("="*80)



            pdf_path = row['Specific Document to test if exact answer required']
            question = row['Questions']
            full_path = pdf_path

            print(f"Document: {pdf_path}")
            print(f"Question: {question}")


            # Extract and process text
            print("\nExtracting and processing text...")

            extracted_text = extract_text_from_json(full_path)
            if not extracted_text:
                raise Exception("No text content found in JSON file")

            # Wrap the extracted text in the expected format
            contract_text = {
                'text': extracted_text,
                'full_response': None,
                'content': {},
                'metadata': {}
            }

            try:
                # JSON reading code
                print("\nSuccessfully extracted text from JSON file")
                print("="*80)
                print("EXTRACTED TEXT CONTENT")
                print("="*80)
                print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
                print("="*80)
            except Exception as e:
                print(f"\nError reading JSON file: {str(e)}")
                raise Exception(f"Failed to extract text from JSON: {str(e)}")

            # Extract text content from the response
            text_content = contract_text.get('text', '')
            if not text_content:
                raise Exception("No text content found in the response")

            chunks = intelligent_chunking(text_content)
            if not chunks:
                raise Exception("No chunks created from document")

            # Insert chunks
            print(f"\nInserting {len(chunks)} chunks...")
            chunk_insert_success = False
            for idx, chunk in enumerate(chunks):
                if insert_text_with_schema(idx, chunk, full_path):
                    chunk_insert_success = True
                else:
                    print(f"Warning: Failed to insert chunk {idx+1}")

            if not chunk_insert_success:
                raise Exception("Failed to insert any chunks")

            # Process question with default values
            print("\nProcessing question...")
            default_response = {
                'intent': {'primary_intent': 'GENERAL'},
                'confidence': {'overall': 0.3},
                'chunks': [],
                'answer': 'No response generated'
            }
            try:
                response = process_question_with_llm_intent(question, text_content, client, VECTOR_MODEL, debug=True)
            except Exception as e:
                print(f"Error in question processing: {str(e)}")
                response = default_response

            # Extract response components with defaults
            intent_info = response.get('intent', {'primary_intent': 'GENERAL'})
            confidence_info = response.get('confidence', {'overall': 0.3})
            chunks_info = response.get('chunks', [])
            answer_text = response.get('answer', 'No response generated')

            # Display results
            print("\nRESULTS:")
            print("-"*40)
            print(f"Intent: {intent_info.get('primary_intent', 'Unknown')}")
            print(f"Confidence: {confidence_info.get('overall', 0):.2f}")

            print("\nRelevant Chunks:")
            print("-"*40)
            for i, chunk in enumerate(chunks_info[1:5]):  # Show top 3 chunks
                print(f"\nChunk {i+1} (Score: {chunk.get('score', 0):.3f}):")
                print(f"Text preview: {chunk.get('text', '')}...")

            print("\nResponse:")
            print("-"*40)
            print(answer_text)

            # Write to CSV
            writer.writerow([
                pdf_path,
                question,
                intent_info.get('primary_intent', 'Unknown'),
                len(chunks_info),
                answer_text
            ])

            print("\nResults written to CSV")
            print("="*80)

        except Exception as e:
            print(f"Error processing document {index + 1}: {str(e)}")
            traceback.print_exc()
            writer.writerow([
                pdf_path if 'pdf_path' in locals() else 'Unknown',
                question if 'question' in locals() else 'Unknown',
                'Error',
                0,
                str(e)
            ])
            continue

print(f"\nAll results have been written to {output_file}")

#------------Insert Chunks to DB----------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def insert_chunks_to_db(contract_text, pdf_path):
    """Insert document chunks into Milvus DB using recursive chunking and key phrase extraction"""
    try:
        print("Processing document for insertion...")

        # Convert text to pages format and get page count
        text_per_page, page_count = (
            contract_text if isinstance(contract_text, tuple)
            else (contract_text.split('\n\n'), 1))
        # Use recursive chunking
        chunks = intelligent_chunking(text_per_page, max_length=500)
        print(f"Created {len(chunks)} chunks using recursive chunking")

        # Extract key phrases from full text
        full_text = " ".join(text_per_page)
        key_phrases = extract_key_phrases(full_text)
        #print(f"Extracted key phrases: {key_phrases}")

        # Prepare metadata
        metadata = {
            "file_name": os.path.basename(pdf_path),
            "page_count": page_count,
            "num_chunks": len(chunks),
            "key_phrases": key_phrases
        }
       # print("Metadata:", metadata)

        # Get embeddings using sentence transformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Prepare data for insertion with chunk-specific key phrases
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_array)):
            # Extract key phrases for each chunk
            chunk_key_phrases = extract_key_phrases(chunk)

            data.append({
                "id": i,
                "vector": embedding.tolist(),
                "file_name": metadata["file_name"],
                "page_count": metadata["page_count"],
                "num_chunks": metadata["num_chunks"],
                "key_phrases": chunk_key_phrases,  # Chunk-specific key phrases
                "text": chunk,  # Store the chunk text
                "doc_key_phrases": metadata["key_phrases"]  # Store document-level key phrases
            })

        # Insert into Milvus
        try:
            client.insert(collection_name=collection_name, data=data)
            print(f"Successfully inserted {len(data)} chunks with metadata")
            return chunks
        except Exception as insert_error:
            print(f"Failed to insert chunks: {str(insert_error)}")
            return []

    except Exception as e:
        print(f"Error in insert_chunks_to_db: {str(e)}")
        return []

# Load spaCy model at the top of the file
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

import spacy
import traceback

def find_phrase_context(text, phrase, context_window=100):
    """Find surrounding context for a phrase"""
    try:
        index = text.lower().find(phrase.lower())
        if index == -1:
            return ""

        start = max(0, index - context_window)
        end = min(len(text), index + len(phrase) + context_window)

        return text[start:end].strip()
    except Exception as e:
        print(f"Error finding context: {str(e)}")
        return ""


def calculate_confidence(intent_info, chunks):
    """Calculate overall confidence score"""
    intent_confidence = float(intent_info.get('confidence', 0))
    chunk_confidence = max(chunk['score'] for chunk in chunks) if chunks else 0

    return {
        "overall": (intent_confidence + chunk_confidence) / 2,
        "intent": intent_confidence,
        "retrieval": chunk_confidence
    }

def simple_keyword_classification(question):
    """Simple keyword-based classification as final fallback"""
    intent_categories = {
        "TERM": ["term", "duration", "period", "expire", "renewal", "start date", "end date"],
        "DOCUMENT": ["document", "agreement", "contract", "titled"],
        "OBLIGATION": ["shall", "must", "required", "obligation", "duty"],
        "RESTRICTION": ["not", "prohibited", "restricted", "limitation"],
        "LIABILITY": ["liable", "responsibility", "warranty", "damages"]
    }

    question_lower = question.lower()
    matches = {
        category: sum(1 for word in words if word in question_lower)
        for category, words in intent_categories.items()
    }

    primary_intent = max(matches.items(), key=lambda x: x[1])[0] if matches else "GENERAL"
    confidence = matches.get(primary_intent, 0) / len(intent_categories[primary_intent]) if primary_intent in intent_categories else 0.3

    return {
        "primary_intent": primary_intent,
        "confidence": confidence,
        "matches": matches
    }

def get_prompt_for_intent(intent_info):
    """Get appropriate prompt based on intent classification"""
    prompts = {
        "TERM": """
            Analyze the contract's term-related information:
            1. Find the effective/start date
            2. Find the termination/end date
            3. Look for specific term clauses

            Focus on dates, periods, and term-related clauses and return these item.
            Provide a clear and structured response with:
            - if ask about term Start date then also check termination date
            - End date or termination date
            """,

        "DOCUMENT": """
            Analyze the document's basic information:
            1. Document type and title
            2. Parties involved
            3. Key dates
            4. Document references

            Provide a clear summary of:
            - Document identification
            - Main parties
            - Important dates
            - Related agreements
            """,

        "OBLIGATION": """
            Analyze the contractual obligations:
            1. Identify key responsibilities
            2. Note mandatory requirements
            3. Find performance criteria
            4. List specific duties

            Focus on terms like "shall", "must", "required to".
            Provide a structured list of obligations.
            """,

        "GENERAL": """
            Analyze the provided content and extract relevant information:
            1. Identify key facts
            2. Note important details
            3. Find specific references
            4. Extract relevant dates and terms

            Provide a clear and direct answer with supporting evidence.
            """
    }

    # Get the appropriate prompt template
    prompt = prompts.get(intent_info["primary_intent"], prompts["GENERAL"])

    return f"""
    {prompt}

    Intent Type: {intent_info["primary_intent"]}
    Confidence: {intent_info["confidence"]:.2f}

    Please provide a clear, structured response that directly addresses the question.
    Include specific quotes or references from the text to support your answer.
    """

# Add configuration section
class Config:
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    COLLECTION_NAME = "demo_collection"
    MODEL_NAME = "qwen2.5:32b"
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Add document classification function
async def classify_document_type(content: str) -> dict:
    """Classify document type using LLM"""
    prompt = """
    Analyze this document content and classify it into the most appropriate category:

    Categories:
    1. AGREEMENT - Main contract or agreement documents
    2. AMENDMENT - Changes or modifications to existing agreements
    3. APPENDIX - Supporting or supplementary documents
    4. SCHEDULE - Time or delivery related documents
    5. OTHER - If none of the above fit

    Content: {content}

    Respond ONLY with a JSON object:
    {{
        "doc_type": "CATEGORY",
        "confidence": 0.0-1.0,
        "reasoning": "Brief explanation"
    }}
    """

    try:
        response = make_ollama_request(prompt.format(content=content[:1000]))  # Use first 1000 chars
        result = json.loads(response)
        return result
    except Exception as e:
        print(f"Error in document classification: {str(e)}")
        return {"doc_type": "OTHER", "confidence": 0.3, "reasoning": "Classification failed"}

def process_and_insert_chunks(chunks, file_name, client, VECTOR_MODEL):
    """Process and insert chunks into Milvus."""
    try:
        vectors = []
        texts = []
        entities_list = []
        ids = []
        current_id = 0

        for chunk in chunks:
            try:
                embedding = VECTOR_MODEL.encode([chunk])[0]
                entities = extract_entities_from_chunk(chunk)

                vectors.append(embedding.tolist())
                texts.append(chunk)
                entities_list.append(json.dumps(entities))
                ids.append(current_id)
                current_id += 1
            except:
                continue

        if not vectors:
            return False

        insert_data = {
            "id": ids,
            "embedding": vectors,
            "text": texts,
            "file_name": [file_name] * len(vectors),
            "upload_date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(vectors),
            "entities": entities_list
        }

        try:
            client.insert(collection_name="demo_collection", data=insert_data)
            return True
        except:
            return False

    except:
        return False
