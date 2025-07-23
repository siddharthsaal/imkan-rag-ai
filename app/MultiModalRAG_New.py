import os
import json
import time
import argparse
import re
from pathlib import Path

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

# Hugging Face Transformers for Re-ranking (even if disabled, class needs it)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


# --- Configuration (MUST MATCH INGESTION SCRIPTS) ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

MILVUS_CHUNKS_COLLECTION_NAME = os.getenv("MILVUS_CHUNKS_COLLECTION_NAME", "text_chunks_milvus")
MILVUS_TABLES_COLLECTION_NAME = os.getenv("MILVUS_TABLES_COLLECTION_NAME", "tabular_data_milvus")
MILVUS_IMAGES_COLLECTION_NAME = os.getenv("MILVUS_IMAGES_COLLECTION_NAME", "sample_image_collection_milvus")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hugging Face Transformers for Re-ranking (conditional import/init)
# These will only be imported and initialized if USE_RERANKER_FOR_RETRIEVAL is True
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers or torch not found. Re-ranking will be disabled.")

TEXT_EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536

LLM_FOR_ANSWER_GENERATION = "gpt-4o"
CONTEXT_ANALYSIS_LLM = "gpt-3.5-turbo"
ROUTER_LLM = "gpt-3.5-turbo"

# --- Re-ranker Configuration ---
USE_RERANKER_FOR_RETRIEVAL = False # Set to False for this simplified pipeline
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MultiModalRAG Class ---
class MultiModalRAG:
    """
    Orchestrates a multi-modal RAG pipeline across text, table, and image data
    stored in Milvus, incorporating conversation awareness and intelligent routing,
    WITHOUT an explicit re-ranking step after initial Milvus retrieval.
    """

    #def __init__(self, config: dict):
    #    self.config = config

    #    self.openai_client = OpenAI(api_key=self.config['openai_api_key'])

    #    if self.config.get('use_reranker_for_retrieval', False):
    #        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.config['reranker_model_name'])
    #        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.config['reranker_model_name']).to(self.config['reranker_device'])
    #        self.reranker_model.eval()
    #        print(f"Re-ranker model '{self.config['reranker_model_name']}' loaded on {self.config['reranker_device']}.")
    #    else:
    #        self.reranker_tokenizer = None
    #        self.reranker_model = None
    #        print("Re-ranking is disabled.")

     #   self.chunks_collection = None
     #   self.tables_collection = None
     #   self.images_collection = None
     #   self.summaries_collection = None
     #   self.collections_loaded = {}

    
    def __init__(self, config: dict):
        self.config = config

        # Validate essential configs
        if not self.config.get('openai_api_key'):
            raise ValueError("OPENAI_API_KEY is not set in config.")
        if not self.config.get('milvus_host') or not self.config.get('milvus_port'):
            raise ValueError("Milvus host and port must be set in config.")

        self.openai_client = OpenAI(api_key=self.config['openai_api_key'])

        # Initialize Re-ranker components if enabled
        self.reranker_tokenizer = None
        self.reranker_model = None
        if self.config.get('use_reranker_for_retrieval', False):
            if _TRANSFORMERS_AVAILABLE:
                try:
                    print(f"Loading re-ranker model: {self.config['reranker_model_name']} on {self.config['reranker_device']}...")
                    self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.config['reranker_model_name'])
                    self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.config['reranker_model_name']).to(self.config['reranker_device'])
                    self.reranker_model.eval()
                    print("Re-ranker model loaded.")
                except Exception as e:
                    print(f"Error loading re-ranker model: {e}. Re-ranking will be disabled.")
                    self.config['use_reranker_for_retrieval'] = False # Disable if load fails
            else:
                print("Transformers or torch not available. Re-ranking disabled.")
                self.config['use_reranker_for_retrieval'] = False
        else:
            print("Re-ranking is disabled via config.")

        self.chunks_collection = None
        self.tables_collection = None
        self.images_collection = None
        self.summaries_collection = None
        self.collections_loaded = {}

    def _get_openai_embedding(self, text: str, model: str = None):
        if model is None:
            model = self.config['text_embedding_model']

        if not text or not text.strip():
            return None
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except APIError as e:
            print(f"OpenAI API Error getting embedding: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred getting embedding: {e}")
            return None
        
    def _get_document_info_from_path(self, file_full_path: str, entity_data: dict = None, default_name: str = "Unknown Document") -> tuple[str, str, str]:
        """
        Tries to infer document name and page info from the file path,
        optionally using provided entity data for more robust parsing.
        Returns: doc_name, page_info, full_path_used (the input file_full_path)
        """
        print("get_document_info--")
        print(f"  file_full_path: {file_full_path}")
        print(f"  entity_data: {entity_data}") # Debug entity data

        doc_name = default_name
        page_info = "N/A"
        
        # Use existing entity data if available and relevant fields are present
        if entity_data:
            # For text chunks/summaries, check 'doc_name' and 'page_number' directly
            if 'doc_name' in entity_data:
                doc_name = entity_data['doc_name']
            if 'page_number' in entity_data: # For chunks
                page_info = f"Page {entity_data['page_number']}"
            if 'total_pages' in entity_data: # For summaries
                page_info = f"Document ({entity_data['total_pages']} pages)" 
            
            # For images/tables, check 'drawing_title', 'table_name' etc.
            elif 'drawing_title' in entity_data:
                doc_name = entity_data['drawing_title']
            elif 'table_name' in entity_data:
                doc_name = entity_data['table_name']
            
            # Rebuild doc_name heuristic from file_full_path if entity's doc_name is not ideal
            if doc_name == default_name or doc_name.startswith("dummy_"): # If doc_name is still default or a dummy
                path_obj_from_full_path = Path(file_full_path)
                file_name_from_path = path_obj_from_full_path.name
                file_stem_from_path = path_obj_from_full_path.stem

                # Common case for sample_images/sample_tables: folder name becomes doc_name
                if "sample_images" in path_obj_from_full_path.parts:
                    idx = path_obj_from_full_path.parts.index("sample_images")
                    if idx > 0: doc_name = path_obj_from_full_path.parts[idx-1].replace("_", " ") + ".pdf"
                elif "sample_tables" in path_obj_from_full_path.parts:
                    idx = path_obj_from_full_path.parts.index("sample_tables")
                    if idx > 0: doc_name = path_obj_from_full_path.parts[idx-1].replace("_", " ") + ".pdf"
                else: # Fallback to filename
                    doc_name = file_name_from_path

            # Page info from filename if not directly from entity data
            if page_info == "N/A":
                page_match_from_path = re.search(r'[Pp]age_?(\d+)$|(\d+)(?:_\d+)?$', file_stem_from_path)
                if page_match_from_path:
                    page_info = f"Page {page_match_from_path.group(1) or page_match_from_path.group(2)}"

            print("Using entity_data/path_heuristics:")
            print(f"  doc_name: {doc_name}")
            print(f"  page_info: {page_info}")
        else: # Fallback to original simple path parsing if no entity_data or it's not useful
            path_obj = Path(file_full_path)
            file_name = path_obj.name
            file_stem = path_obj.stem

            page_match = re.search(r'[Pp]age_?(\d+)$|(\d+)(?:_\d+)?$', file_stem)
            if page_match:
                if page_match.group(1):
                    page_info = f"Page {page_match.group(1)}"
                elif page_match.group(2):
                    page_info = f"Page {page_match.group(2)}"

            if "input_files" in path_obj.parts:
                name_parts = file_stem.split('_')
                if any(re.match(r'[Pp]age_?\d+', p) for p in name_parts):
                    doc_name = "_".join(part for part in name_parts if not re.match(r'[Pp]age_?\d+', part)) + path_obj.suffix
                    if not doc_name: doc_name = file_name
                else: doc_name = file_name
            
            for part in reversed(path_obj.parts):
                if part.lower() in ["images", "sample_images", "sample_tables", "image", "img", "tables"]:
                    break
                if part.lower().endswith((".pdf", ".docx", ".doc", ".txt")):
                    doc_name = part
                    break
                elif path_obj.stem == file_stem and "input_files" not in path_obj.parts:
                    doc_name = part.replace("_", " ") + ".pdf"
                    break

            if doc_name == "N/A":
                doc_name = file_name
            print("Using path_obj parts:")
            print(f"  doc_name: {doc_name}")
            print(f"  page_info: {page_info}")

        return doc_name, page_info, file_full_path # Always return the original full_path passed in
        

    def _milvus_distance_to_similarity(self, distance: float) -> float:
        return 1.0 - distance

    def _classify_user_query(self, user_input: str) -> dict:
        user_input_lower = user_input.lower().strip()

        metadata_fields_all = [
            "chunk_id", "content", "doc_name", "page_number", "chunk_index", "element_type", "full_path",
            "doc_id", "summary_text", "total_pages", # Text RAG fields
            "table_id", "table_name", "summary", "inferred_schema", "sample_qa_pairs", "raw_table_data_csv", "original_image_path", # Table RAG fields
            "image_path", "drawing_title", "drawing_number", "discipline", "drawing_type", "high_level_summary", "key_elements_present", "all_discernible_text_snippets", "generated_qna_pairs" # Image RAG fields
        ]
        
        metadata_expression_pattern = re.compile(
            r"(?:" + "|".join(re.escape(f) for f in metadata_fields_all) + r")\s*(?:==|!=|like|>|<|>=|<=|in)\s*(?:\"[^\"]+\"|'[^']+'|\[[^\]]+\]|\w+)",
            re.IGNORECASE
        )

        split_keywords = [" and ", " where ", " with filter "]
        for keyword in split_keywords:
            if keyword in user_input_lower:
                parts = user_input.split(keyword, 1)
                semantic_part = parts[0].strip()
                potential_metadata_part = parts[1].strip()

                if metadata_expression_pattern.search(potential_metadata_part):
                    print(f"Classification: Detected 'Filtered Semantic' query. Semantic: '{semantic_part}', Filter: '{potential_metadata_part}'")
                    return {
                        "type": "filtered_semantic",
                        "semantic_query_text": semantic_part,
                        "metadata_filter_expr": potential_metadata_part
                    }
        
        if metadata_expression_pattern.fullmatch(user_input):
            print(f"Classification: Detected 'Metadata Only' query. Filter: '{user_input}'")
            return {
                "type": "metadata_only",
                "metadata_query_expr": user_input
            }

        print(f"Classification: Detected 'Pure Semantic' query. Semantic: '{user_input}')")
        return {
            "type": "semantic",
            "semantic_query_text": user_input
        }

    def _analyze_conversation_context(self, current_query: str, conversation_history: list) -> dict:
        if not conversation_history:
            return {"is_conversational": False, "relevant_history_summary": ""}

        history_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])

        system_prompt = (
            "You are an AI assistant that analyzes user queries in the context of past conversations. "
            "Your task is to determine if the 'Current User Query' depends on the 'Past Conversation History' "
            "for its full understanding. If it does, summarize the *key relevant points* from the history "
            "that are necessary to understand the current query. "
            "Output your response strictly as a JSON object with two keys: "
            "'is_conversational' (boolean: true if it depends on history, false otherwise) and "
            "'relevant_history_summary' (string: the summary if conversational, or empty string if not)."
            "Do NOT include any additional text outside the JSON object."
        )

        user_prompt = (
            f"Past Conversation History (last {len(conversation_history)} turns):\n```\n{history_str}\n```\n\n"
            f"Current User Query: \"{current_query}\"\n\n"
            f"Does the 'Current User Query' require understanding of the 'Past Conversation History' to make sense?"
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['context_analysis_llm'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            llm_response_content = response.choices[0].message.content.strip()
            analysis_result = json.loads(llm_response_content)
            return analysis_result
        except APIError as e:
            print(f"OpenAI API Error during conversation analysis: {e}")
            return {"is_conversational": False, "relevant_history_summary": f"Error in context analysis: {e}"}
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error from context analysis LLM: {e}\nRaw content: {llm_response_content[:200]}...")
            return {"is_conversational": False, "relevant_history_summary": "Error: Invalid JSON from context analysis."}
        except Exception as e:
            print(f"An unexpected error occurred during context analysis: {e}")
            return {"is_conversational": False, "relevant_history_summary": f"Unexpected error in context analysis: {e}"}

    # _rerank_documents is kept for consistency in structure but will not be called
    # if USE_RERANKER_FOR_RETRIEVAL is False.
    def _rerank_documents(self, query: str, documents: list) -> list:
        if not self.config.get('use_reranker_for_retrieval', False) or self.reranker_model is None:
            raise RuntimeError("Re-ranking is disabled in config but _rerank_documents was called.")

        if not documents:
            return []

        print(f"  - Re-ranking {len(documents)} candidate chunks with {self.config['reranker_model_name']}...")
        
        # Prepare content for re-ranker, handling different 'content' keys
        docs_content_for_reranker = []
        for doc in documents:
            if 'content' in doc: # For text chunks
                docs_content_for_reranker.append(doc['content'])
            elif 'summary' in doc: # For tables (summary)
                docs_content_for_reranker.append(doc['summary'])
            elif 'high_level_summary' in doc: # For images (high_level_summary)
                docs_content_for_reranker.append(doc['high_level_summary'])
            elif 'drawing_title' in doc: # Fallback for image title
                docs_content_for_reranker.append(doc['drawing_title'])
            else:
                docs_content_for_reranker.append("") # Fallback for missing

        features = self.reranker_tokenizer([query] * len(documents), 
                                     docs_content_for_reranker, # Use the extracted content
                                     padding=True, truncation=True, return_tensors='pt')
        
        features = {k: v.to(self.config['reranker_device']) for k, v in features.items()}

        with torch.no_grad():
            scores = self.reranker_model(**features).logits.squeeze().cpu().numpy()

        if scores.ndim == 0:
            scores = np.array([scores.item()])

        scored_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        print(f"  - Re-ranking complete. Top re-ranker score: {scored_documents[0][1]:.4f}" if scored_documents else "  - No documents to re-rank.")
        return scored_documents

    def _route_query_with_llm(self, query: str, text_context: str = "", table_context: str = "", image_context: str = "") -> dict:
        """
        Uses an LLM to decide which RAG pipeline (text, table, image) is most suitable for the query,
        considering high-level relevant content from each modality.
        """
        system_prompt = (
            "You are an intelligent routing agent for a multi-modal RAG system. "
            "Your task is to determine which data source is most likely to contain the answer to the user's query. "
            "Here are the available data sources and what they contain:\n"
            "- **text**: General text documents, manuals, reports, specifications. Good for conceptual understanding, procedures, definitions, long-form content.\n"
            "- **table**: Tabular data extracted from images, like data sheets, property tables, schedules. Good for specific numerical values, comparative data, structured facts.\n"
            "- **image**: Drawings, diagrams, blueprints, architectural plans (non-tabular). Good for visual information, layouts, components, specific visual elements, Q&A generated from drawings.\n\n"
            "Analyze the user's query and the high-level content preview from each data source below. Decide which single pipeline is the best fit. "
            "Output your decision as a JSON object with 'pipeline_type' (string: 'text', 'table', or 'image') "
            "and 'confidence_score' (float: 0.0 to 1.0, indicating how confident you are in this routing decision, where 1.0 is very confident). "
            "Do NOT include any other text outside the JSON object."
        )

        user_prompt_parts = [f"User Query: \"{query}\"\n\n"]
        user_prompt_parts.append("Here's a high-level overview of the most relevant content found in each data source type for this query:")
        user_prompt_parts.append(f"- Text Documents Preview: {text_context if text_context else 'No highly relevant text document previews found.'}")
        user_prompt_parts.append(f"- Data Tables Preview: {table_context if table_context else 'No highly relevant data table previews found.'}")
        user_prompt_parts.append(f"- Image Drawings Preview: {image_context if image_context else 'No highly relevant image drawing previews found.'}")
        user_prompt_parts.append("\nConsidering the user's query and these high-level descriptions, which single pipeline is the best fit?")
        final_user_prompt = "\n".join(user_prompt_parts)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['router_llm'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            llm_response_content = response.choices[0].message.content.strip()
            decision = json.loads(llm_response_content)
            
            pipeline_type = decision.get("pipeline_type", "text").lower()
            if pipeline_type not in ["text", "table", "image"]:
                pipeline_type = "text"

            confidence = float(decision.get("confidence_score", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            print(f"  - Router LLM Decision: '{pipeline_type}' with confidence {confidence:.2f}")
            return {"pipeline_type": pipeline_type, "confidence_score": confidence}
        except Exception as e:
            print(f"Error in router LLM decision: {e}. Defaulting to 'text'.")
            return {"pipeline_type": "text", "confidence_score": 0.1}

    def generate_llm_answer_for_text_data(
        self,
        original_query: str,
        retrieved_summary: dict,
        relevant_chunks: list, # Changed to list of relevant chunks
        retrieval_score_overall: float, # Overall score for the primary answer/top chunk
        conversation_context: str = ""
    ) -> dict:
        final_response_json = {
            "query": original_query,
            "answer": "No relevant information found or error in generation.",
            "score": float(retrieval_score_overall),
            "sources": [],
            "debug_info": {"retrieved_summary_ids": [], "retrieved_chunk_ids": []},
            "follow_up_questions": [],
            "message": "Answer generation failed."
        }

        if not retrieved_summary and not relevant_chunks:
            return final_response_json

        context_parts = []
        if conversation_context:
            context_parts.append(f"--- Conversation Context ---\n{conversation_context}\n--- End Conversation Context ---")

        if retrieved_summary:
            context_parts.append(f"Document Summary (ID: {retrieved_summary.get('doc_id')} - {retrieved_summary.get('doc_name')}):")
            context_parts.append(retrieved_summary.get('summary_text', 'N/A'))
            context_parts.append(f"Total Pages: {retrieved_summary.get('total_pages', 'N/A')}")
            context_parts.append("-" * 20)
            final_response_json["debug_info"]["retrieved_summary_ids"].append(retrieved_summary.get('doc_id'))

        if relevant_chunks:
            context_parts.append("Relevant Document Chunks:")
            for chunk in relevant_chunks:
                context_parts.append(f"--- Chunk ID: {chunk.get('chunk_id')}, Page: {chunk.get('page_number')}, Type: {chunk.get('element_type')} ---")
                context_parts.append(chunk.get('content', 'N/A'))
                final_response_json["debug_info"]["retrieved_chunk_ids"].append(chunk.get('chunk_id'))
        
        context_for_llm = "\n\n".join(context_parts)

        system_prompt = (
            "You are an AI assistant specialized in extracting and synthesizing information from text documents. "
            "Your primary goal is to provide a single, direct, and concise answer to the user's query, "
            "formatted as a JSON object. "
            "If 'Conversation Context' is provided, use it to understand the user's intent, but *strictly base your final answer only on the 'Retrieved Document' context*. "
            "Strictly adhere to the JSON output format provided below. "
            "If the context contains a specific numerical value, term, or phrase asked in the query, provide it exactly. "
            "For the 'relevant_excerpt' part of the JSON, if the answer comes from a specific section/value, "
            "identify the most relevant 'column' (a descriptive label for what the answer represents, e.g., 'Installation Step', 'Definition of X') "
            "and its 'value' (the specific answer data point from the text chunk). Otherwise, use 'N/A'. "
            "The 'text_chunk_content' should be the content of the chunk that contributed most to the answer."
            "If the context does not contain enough information to fully answer the question, state that in the 'answer' field and use 'N/A' for relevant_column/value."
            "Do NOT include any additional text outside the JSON object."
            "Also, generate 2-3 concise, natural language follow-up questions that a user might ask based on the provided context and the answer given. Ensure these are short and natural language questions."
        )

        llm_output_json_format_template = {
            "answer": "...",
            "relevant_column": "...",
            "relevant_value": "...",
            "follow_up_questions": ["...", "..."]
        }

        user_prompt = (
            f"Original User Query: {original_query}\n\n"
            f"Here is the most relevant information retrieved from the database:\n\n"
            f"{context_for_llm}\n\n"
            f"Based *only* on the above context, provide your response in the following JSON format:\n"
            f"```json\n{json.dumps(llm_output_json_format_template, indent=4)}\n```\n"
            f"Fill in the 'answer' field with a concise answer to the 'Original User Query'. "
            f"If the answer refers to a specific data point from the text, "
            f"identify the 'relevant_column' (a descriptive label for that data point) "
            f"and its 'relevant_value' (the specific answer). "
            "If a specific column/value isn't directly relevant or extractable, use 'N/A' for those fields."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['llm_for_answer_generation'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            llm_response_content = response.choices[0].message.content.strip()
            llm_parsed_json = json.loads(llm_response_content)

            final_response_json["answer"] = llm_parsed_json.get("answer", "N/A")
            final_response_json["follow_up_questions"] = llm_parsed_json.get("follow_up_questions", [])
            
            sources_list = []
            if retrieved_summary:
                doc_name, page_info_summary, full_path_summary = self._get_document_info_from_path(retrieved_summary.get('full_path', 'N/A'), retrieved_summary)
                sources_list.append({
                    "document_name": doc_name,
                    "page_info": page_info_summary,
                    "relevant_excerpt": {
                        "text_chunk_content": retrieved_summary.get('summary_text', 'N/A'),
                        "column": "N/A", "value": "N/A",
                        "content_type": "summary",
                        "key_elements_present": []
                    },
                    "retrieval_id": retrieved_summary.get('doc_id', 'N/A'),
                    "full_path": full_path_summary, # Updated to use correct full_path
                    "score": retrieved_summary.get('milvus_score', 0.0)
                })

            if relevant_chunks:
                for chunk in relevant_chunks:
                    doc_name_chunk, page_info_chunk, full_path_chunk = self._get_document_info_from_path(chunk.get('full_path', 'N/A'), chunk)
                    
                    relevant_excerpt_column = llm_parsed_json.get("relevant_column", "N/A")
                    relevant_excerpt_value = llm_parsed_json.get("relevant_value", "N/A")
                    
                    if isinstance(relevant_excerpt_value, (np.float32, np.float64)):
                        relevant_excerpt_value = float(relevant_excerpt_value)

                    sources_list.append({
                        "document_name": doc_name_chunk,
                        "page_info": f"Page {chunk.get('page_number', 'N/A')}",
                        "relevant_excerpt": {
                            "text_chunk_content": chunk.get('content', 'N/A'),
                            "column": relevant_excerpt_column,
                            "value": relevant_excerpt_value,
                            "content_type": "text_chunk",
                            "key_elements_present": [],
                            "chunk_id": chunk.get('chunk_id', 'N/A'), # Already existed
                            "page_number": chunk.get('page_number', 'N/A') # Already existed
                        },
                        "retrieval_id": chunk.get('chunk_id', 'N/A'),
                        "full_path": full_path_chunk, # Updated to use correct full_path
                        "score": chunk.get('milvus_score', 0.0)
                    })
            
            final_response_json["sources"] = sources_list
            final_response_json["message"] = "Answer successfully generated."

        except APIError as e:
            print(f"OpenAI API Error during answer generation: {e}")
            final_response_json["answer"] = f"Sorry, I encountered an OpenAI API error generating the answer."
            final_response_json["message"] = f"API Error: {e}"
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error from LLM response: {e}\nRaw content: {llm_response_content[:500]}...")
            final_response_json["answer"] = f"Sorry, the AI response was not in the expected JSON format."
            final_response_json["message"] = f"JSON Parsing Error: {e}"
        except Exception as e:
            print(f"An unexpected error occurred during answer generation: {e}")
            final_response_json["answer"] = f"Sorry, I encountered an unexpected error while trying to answer your question."
            final_response_json["message"] = f"Unexpected Error: {e}"
        
        return final_response_json


    def generate_llm_answer_for_image_data(
        self,
        original_query: str,
        retrieved_entity: dict, # entity data
        retrieval_score: float, # score for this entity
        retrieved_image_id: str,
        conversation_context: str = ""
    ) -> dict:
        final_response_json = {
            "query": original_query,
            "answer": "No relevant information found or error in generation.",
            "score": float(retrieval_score),
            "sources": [],
            "debug_info": {"retrieved_image_ids": [retrieved_image_id]},
            "follow_up_questions": [],
            "message": "Answer generation failed."
        }

        if not retrieved_entity:
            return final_response_json

        entity = retrieved_entity
        
        drawing_title = entity.get('drawing_title', 'N/A')
        summary = entity.get('high_level_summary', 'N/A')
        image_path = entity.get('image_path', 'N/A') # This is the full_path for images

        try: key_elements = json.loads(entity.get('key_elements_present', '[]'))
        except json.JSONDecodeError: key_elements = []
        try: qna_pairs = json.loads(entity.get('generated_qna_pairs', '[]'))
        except json.JSONDecodeError: qna_pairs = []

        context_parts = []
        if conversation_context:
            context_parts.append(f"--- Conversation Context ---\n{conversation_context}\n--- End Conversation Context ---")

        context_parts.append(f"Retrieved Document Title: {drawing_title}")
        context_parts.append(f"Associated Image File Path: {image_path}")
        context_parts.append(f"High-Level Summary: {summary}")
        
        if key_elements:
            context_parts.append(f"Key Elements Present: {', '.join(key_elements)}")
        if qna_pairs:
            context_parts.append("Generated Q&A Pairs (Question: Answer format):")
            for qna in qna_pairs:
                if isinstance(qna, dict) and 'question' in qna and 'answer' in qna:
                    context_parts.append(f"- {qna['question']}: {qna['answer']}")

        context_for_llm = "\n".join(context_parts)

        system_prompt = (
            "You are an AI assistant specialized in extracting and synthesizing information from engineering and architectural drawings. "
            "Your goal is to provide a single, direct, and concise answer to the user's query. "
            "If 'Conversation Context' is provided, use it to understand the user's intent, but *strictly base your final answer only on the 'Retrieved Document' context*. "
            "Additionally, identify a 'relevant_column' and 'relevant_value' from the drawing's content (if applicable and specific to the answer). "
            "Strictly adhere to the JSON output format provided below. "
            "If the context does not contain enough information to fully answer the query, state that in the 'answer' field and use 'N/A' for relevant_column/value. "
            "Prioritize factual details from the 'Key Q&A Pairs' and 'High-Level Summary'."
            "Avoid making assumptions or including external knowledge. Focus on what is explicitly stated or can be directly inferred from the provided context."
            "The 'relevant_column' should be a descriptive label, and 'relevant_value' the specific data point related to the answer."
            "For example, if the query is about 'thickness of concrete', and the answer is '76mm', then 'relevant_column' could be 'Thickness' and 'relevant_value' could be '76mm'."
            "Also, from the 'Generated Q&A Pairs' in the context, select 2-3 of the most relevant questions that could serve as follow-up questions. If no relevant Q&A, return empty list."
            "Do NOT include any additional text outside the JSON object."
        )

        llm_output_json_format_template = {
            "answer": "...",
            "relevant_column": "...",
            "relevant_value": "...",
            "follow_up_questions": ["...", "..."]
        }

        user_prompt = (
            f"Original User Query: {original_query}\n\n"
            f"Here is the most relevant information retrieved from the database:\n\n"
            f"{context_for_llm}\n\n"
            f"Based *only* on the above context, provide your response in the following JSON format:\n"
            f"```json\n{json.dumps(llm_output_json_format_template, indent=4)}\n```\n"
            f"Fill in the 'answer' field with a concise answer to the 'Original User Query'. "
            f"If the answer refers to a specific data point from the image (like a dimension, rating, or count), "
            f"identify the 'relevant_column' (e.g., 'Thickness', 'Fire Rating', 'Joint Width') "
            f"and its 'relevant_value' (e.g., '76 mm', '3 hours', '25.5 mm'). "
            f"If a specific column/value isn't directly relevant or extractable, use 'N/A' for those fields."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['llm_for_answer_generation'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            llm_response_content = response.choices[0].message.content.strip()
            llm_parsed_json = json.loads(llm_response_content)

            final_response_json["answer"] = llm_parsed_json.get("answer", "N/A")
            final_response_json["follow_up_questions"] = llm_parsed_json.get("follow_up_questions", [])
            
            doc_name, page_info, full_path_img = self._get_document_info_from_path(image_path, entity) # Updated call

            source_entry = {
                "document_name": doc_name,
                "page_info": page_info,
                "relevant_excerpt": { 
                    "image description": drawing_title,
                    "column": llm_parsed_json.get("relevant_column", "N/A"),
                    "value": llm_parsed_json.get("relevant_value", "N/A"),
                    "content_type": "image",
                    "key_elements_present": key_elements,
                    "sample_qa_pairs": qna_pairs
                },
                "retrieval_id": retrieved_image_id,
                "full_path": full_path_img, # Updated to use correct full_path
                "score": float(retrieval_score)
            }
            final_response_json["sources"].append(source_entry)
            final_response_json["message"] = "Answer successfully generated."

        except APIError as e:
            print(f"OpenAI API Error during answer generation: {e}")
            final_response_json["answer"] = f"Sorry, I encountered an OpenAI API error generating the answer."
            final_response_json["message"] = f"API Error: {e}"
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error from LLM response: {e}\nRaw content: {llm_response_content[:500]}...")
            final_response_json["answer"] = f"Sorry, the AI response was not in the expected JSON format."
            final_response_json["message"] = f"JSON Parsing Error: {e}"
        except Exception as e:
            print(f"An unexpected error occurred during answer generation: {e}")
            final_response_json["answer"] = f"Sorry, I encountered an unexpected error while trying to answer your question."
            final_response_json["message"] = f"Unexpected Error: {e}"
        
        return final_response_json


    def generate_llm_answer_for_table_data(
        self,
        original_query: str, retrieved_entity: dict, retrieval_score: float,
        top_hit_id: str,
        conversation_context: str = ""
    ) -> dict:
        final_response_json = {
            "query": original_query,
            "answer": "No relevant information found or error in generation.",
            "score": float(retrieval_score),
            "sources": [],
            "sample_qa_pairs": [],
            "debug_info": {"retrieved_table_ids": [top_hit_id] if top_hit_id else []},
            "follow_up_questions": [], # Added
            "message": "Answer generation failed."
        }

        if not retrieved_entity:
            return final_response_json

        table_id = retrieved_entity.get('table_id', 'N/A')
        table_name = retrieved_entity.get('table_name', 'N/A')
        original_image_path = retrieved_entity.get('original_image_path', 'N/A')
        summary = retrieved_entity.get('summary', 'N/A')
        raw_table_data_csv = retrieved_entity.get('raw_table_data_csv', 'N/A')

        try: inferred_schema = json.loads(retrieved_entity.get('inferred_schema', '{}'))
        except json.JSONDecodeError: inferred_schema = {}
        try: sample_qa_pairs = json.loads(retrieved_entity.get('sample_qa_pairs', '[]'))
        except json.JSONDecodeError: sample_qa_pairs = []

        context_parts = []
        if conversation_context:
            context_parts.append(f"--- Conversation Context ---\n{conversation_context}\n--- End Conversation Context ---")

        context_parts.append(f"Retrieved Table ID: {table_id}")
        context_parts.append(f"Table Name (from filename): {table_name}")
        context_parts.append(f"Original Image Path: {original_image_path}")
        context_parts.append(f"Table Summary: {summary}")

        if inferred_schema:
            schema_str = ", ".join([f"{col}: {dtype}" for col, dtype in inferred_schema.items()])
            context_parts.append(f"Inferred Table Schema (Column: Type): {schema_str}")
        if sample_qa_pairs:
            context_parts.append("\nGenerated Sample Q&A Pairs from this table:")
            for qna in sample_qa_pairs:
                if isinstance(qna, dict) and 'question' in qna and 'answer' in qna:
                    context_parts.append(f"- {qna['question']}: {qna['answer']}")
        if raw_table_data_csv and raw_table_data_csv != 'N/A':
            context_parts.append("\nComplete Raw Table Data (CSV format):")
            context_parts.append("```csv")
            context_parts.append(raw_table_data_csv)
            context_parts.append("```")

        context_for_llm = "\n".join(context_parts)

        system_prompt = (
            "You are an AI assistant specialized in extracting and synthesizing information from tabular data. "
            "Your primary goal is to provide a single, direct, and concise answer to the user's question, "
            "formatted as a JSON object. "
            "If 'Conversation Context' is provided, use it to understand the user's intent, but *strictly base your final answer only on the 'Retrieved Document' context*. "
            "Strictly adhere to the JSON output format provided below. "
            "If the query asks for a specific value from the table, prioritize extracting exact values from it when asked for specific data points. "
            "For the 'relevant_excerpt' part of the JSON, try to identify the most relevant 'column' and its 'value' from the table for the answer, "
            "otherwise provide 'N/A'. The 'table description' should be the table's summary."
            "If the context does not contain enough information to fully answer the question, state that in the 'answer' field. "
            "Also, from the 'Generated Sample Q&A Pairs' in the context, select 2-3 of the most relevant questions that could serve as follow-up questions. If no relevant Q&A, return empty list."
            "Do NOT include any additional text outside the JSON object."
        )

        llm_output_json_format_template = {
            "answer": "...",
            "relevant_column": "...",
            "relevant_value": "...",
            "follow_up_questions": ["...", "..."]
        }

        user_prompt = (
            f"Original User Query: {original_query}\n\n"
            f"Here is the most relevant information retrieved from the database about a table:\n\n"
            f"{context_for_llm}\n\n"
            f"Based *only* on the above context, provide your response in the following JSON format:\n"
            f"```json\n{json.dumps(llm_output_json_format_template, indent=4)}\n```\n"
            f"Fill in the 'answer' field with a concise answer to the 'Original User Query'. "
            f"If the answer is a specific data point, identify the 'relevant_column' and its 'relevant_value' from the table. "
            f"If a specific column/value isn't directly relevant or extractable, use 'N/A' for those fields."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['llm_for_answer_generation'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            llm_response_content = response.choices[0].message.content.strip()
            llm_parsed_json = json.loads(llm_response_content)

            final_response_json["answer"] = llm_parsed_json.get("answer", "N/A")
            final_response_json["follow_up_questions"] = llm_parsed_json.get("follow_up_questions", [])
            
            doc_name, page_info, full_path_table = self._get_document_info_from_path(original_image_path, retrieved_entity) # Updated call

            source_entry = {
                "document_name": doc_name,
                "page_info": page_info,
                "relevant_excerpt": {
                    "table description": summary,
                    "column": llm_parsed_json.get("relevant_column", "N/A"),
                    "value": llm_parsed_json.get("relevant_value", "N/A"),
                    "content_type": "table",
                    "key_elements_present": [],
                    "sample_qa_pairs": sample_qa_pairs
                },
                "retrieval_id": table_id,
                "full_path": full_path_table, # Updated to use correct full_path
                "score": float(retrieval_score)
            }
            final_response_json["sources"].append(source_entry)
            final_response_json["message"] = "Answer successfully generated."
            final_response_json["sample_qa_pairs"] = sample_qa_pairs

        except APIError as e:
            print(f"OpenAI API Error during answer generation: {e}")
            final_response_json["answer"] = f"Sorry, I encountered an OpenAI API error generating the answer."
            final_response_json["message"] = f"API Error: {e}"
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error from LLM response: {e}\nRaw content: {llm_response_content[:500]}...")
            final_response_json["answer"] = f"Sorry, the AI response was not in the expected JSON format."
            final_response_json["message"] = f"JSON Parsing Error: {e}"
        except Exception as e:
            print(f"An unexpected error occurred during answer generation: {e}")
            final_response_json["answer"] = f"Sorry, I encountered an unexpected error while trying to answer your question."
            final_response_json["message"] = f"Unexpected Error: {e}"
        
        return final_response_json

    def _retrieve_for_text_pipeline(self, query_vector, semantic_query_text, chunks_collection_obj: Collection, summaries_collection_obj: Collection, metadata_filter_expr: str = None):
        """Retrieves top text chunk(s) and its summary based on Milvus similarity."""
        print(f"  - Retrieving best text chunk(s) from '{chunks_collection_obj.name}'...")
        chunk_output_fields = ["chunk_id", "content", "doc_name", "page_number", "chunk_index", "element_type", "full_path"]
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        summary_output_fields = ["doc_id", "summary_text", "doc_name", "full_path", "total_pages"]
        summary_search_results = summaries_collection_obj.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=1,
            expr=metadata_filter_expr,
            output_fields=summary_output_fields
        )

        retrieved_summary = None
        relevant_chunks_with_scores = []
        overall_best_retrieval_score = 0.0

        if summary_search_results and summary_search_results[0]:
            top_summary_hit = summary_search_results[0][0]
            retrieved_summary = top_summary_hit.entity.to_dict()
            retrieved_summary['milvus_score'] = self._milvus_distance_to_similarity(top_summary_hit.distance)
            print(f"  - Top summary found: '{retrieved_summary.get('doc_name')}' (Milvus Distance: {top_summary_hit.distance:.4f}).")

            chunk_filter_expr = f'doc_name == "{retrieved_summary.get("doc_name")}"'
            if metadata_filter_expr:
                chunk_filter_expr = f"({chunk_filter_expr}) and ({metadata_filter_expr})"

            chunk_search_results = chunks_collection_obj.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=3, # Retrieve 3 best chunks for context (adjust as needed)
                expr=chunk_filter_expr,
                output_fields=chunk_output_fields
            )
            
            if chunk_search_results and chunk_search_results[0]:
                for i, hit in enumerate(chunk_search_results[0]):
                    chunk_data = hit.entity.to_dict()
                    chunk_score = self._milvus_distance_to_similarity(hit.distance)
                    chunk_data['milvus_score'] = chunk_score
                    relevant_chunks_with_scores.append(chunk_data)
                    if i == 0:
                        overall_best_retrieval_score = chunk_score
                print(f"  - Found {len(relevant_chunks_with_scores)} relevant chunks. Overall best chunk score: {overall_best_retrieval_score:.4f}.")
            else:
                print("  - No relevant chunks found within the top document.")
        else:
            print("  - No relevant document summaries found for the query.")
        
        return retrieved_summary, relevant_chunks_with_scores, overall_best_retrieval_score

    def _retrieve_best_table(self, query_vector, semantic_query_text, collection_obj: Collection, metadata_filter_expr: str = None):
        """Retrieves top table hit."""
        print(f"  - Retrieving best table from '{collection_obj.name}'...")
        table_output_fields = ["table_id", "table_name", "summary", "original_image_path", "raw_table_data_csv", "inferred_schema", "sample_qa_pairs"]
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        table_search_results = collection_obj.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=1,
            expr=metadata_filter_expr,
            output_fields=table_output_fields
        )

        if not table_search_results or not table_search_results[0]:
            return None, None

        top_hit = table_search_results[0][0]
        top_hit_entity = top_hit.entity.to_dict()
        top_hit_score = self._milvus_distance_to_similarity(top_hit.distance)
        
        top_hit_entity['milvus_score'] = top_hit_score

        print(f"  - Top table found: '{top_hit_entity.get('table_name')}' (Milvus Distance: {top_hit.distance:.4f}).")
        return top_hit_entity, top_hit_score


    def _retrieve_best_image(self, query_vector, semantic_query_text, collection_obj: Collection, metadata_filter_expr: str = None):
        """Retrieves top image hit."""
        print(f"  - Retrieving best image from '{collection_obj.name}'...")
        image_output_fields = ["image_id", "drawing_title", "high_level_summary", "image_path", "discipline", "drawing_type", "drawing_number", "key_elements_present", "all_discernible_text_snippets", "generated_qna_pairs"]
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        image_search_results = collection_obj.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=1,
            expr=metadata_filter_expr,
            output_fields=image_output_fields
        )

        if not image_search_results or not image_search_results[0]:
            return None, None

        top_hit = image_search_results[0][0]
        top_hit_entity = top_hit.entity.to_dict()
        top_hit_score = self._milvus_distance_to_similarity(top_hit.distance)

        top_hit_entity['milvus_score'] = top_hit_score

        print(f"  - Top image found: '{top_hit_entity.get('drawing_title')}' (Milvus Distance: {top_hit.distance:.4f}).")
        return top_hit_entity, top_hit_score


    def _connect_and_load_milvus_collections(self):
        """Connects to Milvus and loads all necessary collections."""
        try:
            connections.connect("default", host=self.config['milvus_host'], port=self.config['milvus_port'],user=self.config['user'],password=self.config['password'])
            print(f"\nConnected to Milvus at {self.config['milvus_host']}:{self.config['milvus_port']}")

            collection_names = {
                "text": self.config['milvus_chunks_collection_name'],
                "table": self.config['milvus_tables_collection_name'],
                "image": self.config['milvus_images_collection_name'],
                "summary": self.config['milvus_summaries_collection_name']
            }

            for key, name in collection_names.items():
                try:
                    if utility.has_collection(name):
                        col_obj = Collection(name)
                        col_obj.load()
                        self.collections_loaded[key] = col_obj
                        print(f"Collection '{name}' loaded.")
                    else:
                        print(f"Warning: Collection '{name}' not found.")
                except Exception as e:
                    print(f"Error loading collection '{name}': {e}")
            
            self.chunks_collection = self.collections_loaded.get("text")
            self.tables_collection = self.collections_loaded.get("table")
            self.images_collection = self.collections_loaded.get("image")
            self.summaries_collection = self.collections_loaded.get("summary")

        except Exception as e:
            print(f"Error connecting to Milvus or loading collections: {e}")
            raise

    def close(self):
        """Releases all loaded Milvus collections and disconnects."""
        for col in self.collections_loaded.values():
            if col:
                try: col.release()
                except Exception as e: print(f"Warning: Could not release collection {col.name}: {e}")
        connections.disconnect("default")
        print("Disconnected from Milvus.")


    def orchestrate_query(self, user_request: dict) -> dict:
        original_user_query = user_request.get("current_query", "")
        conversation_history = user_request.get("conversation_history", [])

        conversation_analysis_result = self._analyze_conversation_context(original_user_query, conversation_history)
        print("Conversational Analysis Result:", conversation_analysis_result)
        is_conversational = conversation_analysis_result["is_conversational"]
        relevant_history_summary = conversation_analysis_result["relevant_history_summary"]

        final_orchestrator_output = {
            "query": original_user_query,
            "answer": "No relevant information found across any source.",
            "score": 0.0,
            "sources": [],
            "debug_info": {
                "conversation_analysis": conversation_analysis_result,
                "routing_strategy_used": "N/A",
                "intelligent_router_decision": "N/A",
                "retrieved_ids_text": [],
                "retrieved_ids_table": [],
                "retrieved_ids_image": []
            },
            "message": "Orchestration process completed."
        }
        
        if not (self.chunks_collection or self.tables_collection or self.images_collection):
            final_orchestrator_output["answer"] = "No content collections found/loaded in Milvus. Cannot perform search."
            final_orchestrator_output["message"] = "Orchestration failed: No collections."
            return final_orchestrator_output

        classification = self._classify_user_query(original_user_query)
        semantic_query_text = classification.get("semantic_query_text")
        metadata_filter_expr = classification.get("metadata_filter_expr")
        is_metadata_only = (classification["type"] == "metadata_only")

        if is_metadata_only:
            print("\n--- Executing Metadata-Only Search Across Modalities ---")
            unified_metadata_results = []
            
            if self.chunks_collection:
                chunk_output_fields = ["chunk_id", "content", "doc_name", "page_number", "full_path"]
                try:
                    results = self.chunks_collection.query(expr=metadata_filter_expr, output_fields=chunk_output_fields, limit=5)
                    for entity in results:
                        # Pass entity to get more robust info if available, and capture full_path from returned tuple
                        full_path_val = entity.get('full_path', 'N/A')
                        doc_name, page_info, _ = self._get_document_info_from_path(full_path_val, entity) 
                        unified_metadata_results.append({
                            "document_name": doc_name,
                            "page_info": f"Page {entity.get('page_number', 'N/A')}", # Prefer entity's page_number for chunk
                            "content_type": "text",
                            "preview_text": entity.get('content', '')[:150] + "..." if len(entity.get('content', '')) > 150 else entity.get('content', ''),
                            "complete_paragraph": entity.get('content', ''),
                            "similarity_score": 0.0,
                            "retrieval_id": entity.get('chunk_id'),
                            "full_path": full_path_val # Use the original full path from entity
                        })
                        final_orchestrator_output["debug_info"]["retrieved_ids_text"].append(entity.get('chunk_id'))
                except milvus_exceptions.MilvusException as e: print(f"Error querying text chunks metadata: {e}")

            if self.tables_collection:
                table_output_fields = ["table_id", "table_name", "summary", "original_image_path"]
                try:
                    results = self.tables_collection.query(expr=metadata_filter_expr, output_fields=table_output_fields, limit=5)
                    for entity in results:
                        # Pass entity to get more robust info if available, and capture full_path from returned tuple
                        original_image_path_val = entity.get('original_image_path', 'N/A')
                        doc_name, page_info, _ = self._get_document_info_from_path(original_image_path_val, entity)
                        unified_metadata_results.append({
                            "document_name": doc_name, "page_info": page_info,
                            "content_type": "table", "preview_text": entity.get('table_name', '') + ": " + entity.get('summary', '')[:100] + "...",
                            "retrieval_id": entity.get('table_id'),
                            "similarity_score": 0.0,
                            "full_path": original_image_path_val # Use the original full path from entity
                        })
                        final_orchestrator_output["debug_info"]["retrieved_ids_table"].append(entity.get('table_id'))
                except milvus_exceptions.MilvusException as e: print(f"Error querying tables metadata: {e}")
            
            if self.images_collection:
                image_output_fields = ["image_id", "drawing_title", "high_level_summary", "image_path"]
                try:
                    results = self.images_collection.query(expr=metadata_filter_expr, output_fields=image_output_fields, limit=5)
                    for entity in results:
                        # Pass entity to get more robust info if available, and capture full_path from returned tuple
                        image_path_val = entity.get('image_path', 'N/A')
                        doc_name, page_info, _ = self._get_document_info_from_path(image_path_val, entity)
                        unified_metadata_results.append({
                            "document_name": doc_name, "page_info": page_info,
                            "content_type": "image", "preview_text": entity.get('drawing_title', '') + ": " + entity.get('high_level_summary', '')[:100] + "...",
                            "retrieval_id": entity.get('image_id'),
                            "similarity_score": 0.0,
                            "full_path": image_path_val # Use the original full path from entity
                        })
                        final_orchestrator_output["debug_info"]["retrieved_ids_image"].append(entity.get('image_id'))
                except milvus_exceptions.MilvusException as e: print(f"Error querying images metadata: {e}")
            
            final_orchestrator_output["answer"] = f"Found {len(unified_metadata_results)} items matching your metadata query."
            final_orchestrator_output["results"] = unified_metadata_results
            final_orchestrator_output["message"] = "Metadata search completed."
            final_orchestrator_output["debug_info"]["routing_strategy_used"] = "Metadata-Only Search"
            return final_orchestrator_output


        # Handle Semantic Queries (Orchestration logic - now unified search)
        query_vector = self._get_openai_embedding(semantic_query_text)
        if query_vector is None:
            final_orchestrator_output["answer"] = "Failed to generate embedding for query. Cannot perform semantic search."
            final_orchestrator_output["message"] = "Embedding generation failed."
            return final_orchestrator_output
        
        router_query_text = original_user_query
        if is_conversational and relevant_history_summary:
            router_query_text = f"User query: '{original_user_query}'. Conversation history context: '{relevant_history_summary}'"

        # --- High-level content retrieval for router context ---
        text_preview_content = "No relevant content found."
        table_preview_content = "No relevant content found."
        image_preview_content = "No relevant content found."

        search_params_for_preview = {"metric_type": "COSINE", "params": {"nprobe": 5}}

        if self.summaries_collection:
            summary_preview_fields = ["summary_text", "doc_name"]
            summary_preview_results = self.summaries_collection.search(
                data=[query_vector], anns_field="embedding", param=search_params_for_preview, limit=1, output_fields=summary_preview_fields
            )
            if summary_preview_results and summary_preview_results[0]:
                top_summary = summary_preview_results[0][0].entity.to_dict()
                text_preview_content = f"Top text document summary: '{top_summary.get('summary_text', '')}' from '{top_summary.get('doc_name', '')}'."

        if self.tables_collection:
            table_preview_fields = ["table_name", "summary"]
            table_preview_results = self.tables_collection.search(
                data=[query_vector], anns_field="embedding", param=search_params_for_preview, limit=1, output_fields=table_preview_fields
            )
            if table_preview_results and table_preview_results[0]:
                top_table = table_preview_results[0][0].entity.to_dict()
                table_preview_content = f"Top table summary: '{top_table.get('summary', '')}' from '{top_table.get('table_name', '')}'."

        if self.images_collection:
            image_preview_fields = ["drawing_title", "high_level_summary"]
            image_preview_results = self.images_collection.search(
                data=[query_vector], anns_field="embedding", param=search_params_for_preview, limit=1, output_fields=image_preview_fields
            )
            if image_preview_results and image_preview_results[0]:
                top_image = image_preview_results[0][0].entity.to_dict()
                image_preview_content = f"Top image drawing: '{top_image.get('drawing_title', '')}' summarized as '{top_image.get('high_level_summary', '')}'."

        # --- Intelligent Router Decision (The core of this simpler pipeline) ---
        print("\n--- Intelligent Router Decision ---")
        # Removed explicit _route_query_with_llm call here, as it's not the primary routing strategy now.
        # The routing decision is now implicit through the "search all & pick best" approach.
        final_orchestrator_output["debug_info"]["intelligent_router_decision"] = "N/A (Search All Modalities)"


        # --- Strategy: Query All 3 RAGs & Cross-Modal Re-rank ---
        print("\n--- Performing Unified Semantic Search Across All Modalities ---")
        all_candidate_hits_for_reranking = []

        # Retrieve best candidate from TEXT pipeline
        if self.chunks_collection and self.summaries_collection:
            # _retrieve_for_text_pipeline returns retrieved_summary, relevant_chunks (list), overall_best_score
            retrieved_summary_text, relevant_chunks_text, milvus_score_text_pipeline = self._retrieve_for_text_pipeline(
                query_vector, semantic_query_text, self.chunks_collection, self.summaries_collection, metadata_filter_expr
            )
            if retrieved_summary_text and relevant_chunks_text:
                # For cross-modal re-ranking, we typically use the content of the best chunk.
                # relevant_chunks_text[0] is the best chunk.
                best_chunk_for_cross_rerank = relevant_chunks_text[0]
                # Attach summary data to the best chunk, so the LLM generation function gets it
                best_chunk_for_cross_rerank['retrieved_summary_data'] = retrieved_summary_text 
                
                all_candidate_hits_for_reranking.append({
                    "type": "text",
                    "content": best_chunk_for_cross_rerank.get("content", ""), # Re-ranker needs raw content
                    "milvus_hit_data": best_chunk_for_cross_rerank, # Original Milvus data (now includes summary)
                    "milvus_score": milvus_score_text_pipeline # Score for the text pipeline overall
                })
                # Populate debug info for text (all retrieved chunks/summary for this path)
                final_orchestrator_output["debug_info"]["retrieved_ids_text"].append(retrieved_summary_text.get('doc_id'))
                for chunk in relevant_chunks_text:
                    final_orchestrator_output["debug_info"]["retrieved_ids_text"].append(chunk.get('chunk_id'))


        # Retrieve best candidate from TABLE pipeline
        if self.tables_collection:
            top_table_entity, table_score = self._retrieve_best_table(query_vector, semantic_query_text, self.tables_collection, metadata_filter_expr)
            if top_table_entity:
                all_candidate_hits_for_reranking.append({
                    "type": "table",
                    "content": top_table_entity.get("summary", "") or top_table_entity.get("table_name", ""), # Use summary or name for re-ranker content
                    "milvus_hit_data": top_table_entity, # Original Milvus data
                    "milvus_score": table_score # Score from table pipeline retrieval
                })
                final_orchestrator_output["debug_info"]["retrieved_ids_table"].append(top_table_entity.get('table_id'))

        # Retrieve best candidate from IMAGE pipeline
        if self.images_collection:
            top_image_entity, image_score = self._retrieve_best_image(query_vector, semantic_query_text, self.images_collection, metadata_filter_expr)
            if top_image_entity:
                all_candidate_hits_for_reranking.append({
                    "type": "image",
                    "content": top_image_entity.get("high_level_summary", "") or top_image_entity.get("drawing_title", ""), # Use summary or title for re-ranker content
                    "milvus_hit_data": top_image_entity, # Original Milvus data
                    "milvus_score": image_score # Score from image pipeline retrieval
                })
                final_orchestrator_output["debug_info"]["retrieved_ids_image"].append(top_image_entity.get('image_id'))

        best_overall_result = None
        if all_candidate_hits_for_reranking:
            if self.config.get('use_reranker_for_retrieval', False): # Use re-ranker if enabled
                print("\n--- Performing Cross-Modal Re-ranking on candidates ---")
                cross_reranked_scores = self._rerank_documents(router_query_text, all_candidate_hits_for_reranking)
                if cross_reranked_scores:
                    best_overall_candidate = cross_reranked_scores[0][0] # The candidate dict
                    best_overall_score = float(cross_reranked_scores[0][1]) # The re-ranker score
                    print(f"  - Cross-modal re-ranker selected top hit of type '{best_overall_candidate['type']}' (Score: {best_overall_score:.4f}).")
                    best_overall_result = {"candidate": best_overall_candidate, "score": best_overall_score}
                else:
                    print("  - Cross-modal re-ranking failed to yield results. Falling back to highest Milvus score.")
            
            # Fallback to highest Milvus score if re-ranker is disabled or failed
            if not best_overall_result and all_candidate_hits_for_reranking:
                print("  - Selecting best candidate based on highest Milvus similarity score.")
                sorted_by_milvus_score = sorted(all_candidate_hits_for_reranking, key=lambda x: x["milvus_score"], reverse=True)
                best_overall_candidate = sorted_by_milvus_score[0]
                best_overall_score = best_overall_candidate["milvus_score"]
                print(f"  - Top Milvus candidate of type '{best_overall_candidate['type']}' (Score: {best_overall_score:.4f}).")
                best_overall_result = {"candidate": best_overall_candidate, "score": best_overall_score}
        
        # --- Generate Final Answer from the Best Overall Result ---
        if best_overall_result:
            chosen_candidate_type = best_overall_result['candidate']['type']
            chosen_milvus_data = best_overall_result['candidate']['milvus_hit_data']
            chosen_score = best_overall_result['score']
            
            print(f"\n--- Generating Final Answer via {chosen_candidate_type.upper()} RAG Pipeline ---")
            
            final_answer_llm_output = None
            if chosen_candidate_type == "text":
                # For text, chosen_milvus_data is the best chunk, and it contains 'retrieved_summary_data'
                retrieved_summary_for_llm = chosen_milvus_data.get('retrieved_summary_data')
                top_relevant_chunk_for_llm = chosen_milvus_data # The best chunk itself
                
                final_answer_llm_output = self.generate_llm_answer_for_text_data(
                    original_user_query, retrieved_summary_for_llm, [top_relevant_chunk_for_llm], # Pass as list
                    chosen_score, relevant_history_summary if is_conversational else ""
                )
            elif chosen_candidate_type == "table":
                final_answer_llm_output = self.generate_llm_answer_for_table_data(
                    original_user_query, chosen_milvus_data, chosen_score, chosen_milvus_data.get('table_id'), relevant_history_summary if is_conversational else ""
                )
            elif chosen_candidate_type == "image":
                final_answer_llm_output = self.generate_llm_answer_for_image_data(
                    original_user_query, chosen_milvus_data, chosen_score, chosen_milvus_data.get('image_id'), relevant_history_summary if is_conversational else ""
                )

            if final_answer_llm_output:
                final_orchestrator_output.update(final_answer_llm_output)
                final_orchestrator_output["debug_info"]["routing_strategy_used"] = f"Search All Modalities ({chosen_candidate_type.upper()})"
                final_orchestrator_output["debug_info"]["conversation_analysis"] = conversation_analysis_result
            else:
                final_orchestrator_output["answer"] = f"Failed to generate answer from the best '{chosen_candidate_type}' result."
                final_orchestrator_output["message"] = "Answer generation failed for best candidate."
                final_orchestrator_output["score"] = 0.0

        else:
            final_orchestrator_output["answer"] = "No relevant information found across any source to generate an answer."
            final_orchestrator_output["message"] = "No relevant data found."
            final_orchestrator_output["score"] = 0.0

        return final_orchestrator_output