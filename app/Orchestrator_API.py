import os
import json
from openai import OpenAI
import os
import base64
import json
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from pathlib import Path # For easier path handling
import MultiModalRAG_New

import argparse


# Milvus specific imports
from pymilvus import (
    connections,
    utility,
    Collection,
    exceptions as milvus_exceptions
)

# OpenAI specific imports
from openai import OpenAI
from openai import APIError

# Hugging Face Transformers for Re-ranking
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np



# Collection names from your previous ingestion scripts
#MILVUS_CHUNKS_COLLECTION_NAME = "pdf_text_first_iteration"
#MILVUS_TABLES_COLLECTION_NAME = "pdf_table_first_iteration"
#MILVUS_IMAGES_COLLECTION_NAME = "pdf_image_first_iteration"
#MILVUS_SUMMARIES_COLLECTION_NAME = "text_rag_summary_first_iteration"


from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
import json
import os
import uvicorn

import json

#http://192.168.171.130:30986/

from dotenv import load_dotenv

# Load .env values
load_dotenv()

# --- Re-ranker Configuration ---
USE_RERANKER_FOR_RETRIEVAL = True # Whether to use re-ranker after Milvus for candidate selection
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Re-ranker Model Initialization ---
print(f"Loading re-ranker model: {RERANKER_MODEL_NAME} on {RERANKER_DEVICE}...")
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(RERANKER_DEVICE)
reranker_model.eval()
print("Re-ranker model loaded.")

# --- Configuration dictionary ---
    # Make sure these match your Milvus setup and OpenAI models from ingestion scripts
# Read env variables
rag_config = {
    "milvus_host": os.getenv("MILVUS_HOST"),
    "milvus_port": os.getenv("MILVUS_PORT"),
    "user": os.getenv("MILVUS_USER"),
    "password": os.getenv("MILVUS_PASSWORD"),
    "milvus_chunks_collection_name": os.getenv("MILVUS_CHUNKS_COLLECTION_NAME"),
    "milvus_tables_collection_name": os.getenv("MILVUS_TABLES_COLLECTION_NAME"),
    "milvus_images_collection_name": os.getenv("MILVUS_IMAGES_COLLECTION_NAME"),
    "milvus_summaries_collection_name": os.getenv("MILVUS_SUMMARIES_COLLECTION_NAME"),
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "text_embedding_model": os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-small"),
    "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", 3072)),
    "llm_for_answer_generation": os.getenv("LLM_ANSWER", "gpt-4o"),
    "context_analysis_llm": os.getenv("LLM_CONTEXT", "gpt-4o"),
    "router_llm": os.getenv("LLM_ROUTER", "gpt-3.5-turbo"),
    "use_reranker_for_retrieval": os.getenv("USE_RERANKER", "False") == "True",
    "reranker_model_name": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "reranker_device": "cuda" if torch.cuda.is_available() else "cpu"
}



# Initialize FastAPI app
app = FastAPI(title="RAG Orchestrator API")

def process_rag_json(data):
    result = {
        "query": data.get("query"),
        "answer": data.get("answer"),
        #"score": data.get("score"),
        "message": data.get("message"),
        "follow_up_questions": data.get("follow_up_questions", []),
        "sources": []
    }

    retrieved_chunks = data.get("debug_info", {}).get("retrieved_chunk_ids", [])

    for i, source in enumerate(data.get("sources", [])):
        result["sources"].append({
            "retrieved_chunk_id": retrieved_chunks[i] if i < len(retrieved_chunks) else None,
            "document_name": source.get("document_name"),
            "text_chunk_content": source.get("relevant_excerpt", {}).get("text_chunk_content"),
            "page_info": source.get("page_info"),
            "score":source.get("score"),
            "content_type": source.get("relevant_excerpt", {}).get("content_type")
        })

    return result


class ChatRequest(BaseModel):
    current_query: str
    conversation_history: List[dict] = []


@app.get("/imkan_rag", summary="Query the RAG pipeline")
def query_rag(user_query: str = Query(..., description="User's input query")):
    try:
        # Build request dict
        user_request = {
            "current_query": user_query,
            "conversation_history": []
        }

        # Init orchestrator and load collections
        rag_orchestrator = MultiModalRAG_New.MultiModalRAG(rag_config)
        rag_orchestrator._connect_and_load_milvus_collections()

        # Call orchestrator
        final_response = rag_orchestrator.orchestrate_query(user_request)

        # Post-process the response
        processed_json = process_rag_json(final_response)

        # Cleanup
        rag_orchestrator.close()

        return {"processed_json": processed_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", 7501))

