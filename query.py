"""
Hybrid RAG Query System
Combines BM25 and Vector Search for optimal retrieval
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Environment and configuration
from dotenv import load_dotenv

# Database
import psycopg2
from psycopg2.extras import RealDictCursor

# Ollama
import ollama

# BM25
from rank_bm25 import BM25Okapi
import numpy as np

# NLP
import nltk
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'query_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HybridRAG:
    """Hybrid RAG system combining BM25 and Vector Search"""

    def __init__(self):
        """Initialize the Hybrid RAG system"""
        logger.info("Initializing HybridRAG system...")

        # Load environment variables
        load_dotenv()

        # PostgreSQL configuration
        self.db_config = {
            'dbname': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432')
        }
        self.table_name = os.getenv('VECTOR_TABLE_NAME', 'documents')

        # Embedding configuration
        self.embed_model = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text:latest')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        # LLM configuration
        self.llm_model = os.getenv('LLM_MODEL_NAME', 'llama2')
        self.llm_provider = os.getenv('LLM_PROVIDER', 'ollama')

        # RAG settings
        self.top_k = int(os.getenv('TOP_K', '7'))

        logger.info(f"Database: {self.db_config['dbname']}, Table: {self.table_name}")
        logger.info(f"LLM Model: {self.llm_model}, Top K: {self.top_k}")

        # Download required NLTK data
        self._download_nltk_data()

        # Cache for BM25
        self.bm25_corpus = None
        self.bm25_model = None
        self.corpus_docs = None

        logger.info("HybridRAG system initialized successfully")

    def _download_nltk_data(self):
        """Download required NLTK data packages"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

    def embed_query(self, query: str) -> List[float]:
        """Generate embeddings for query using Ollama"""
        logger.info("Generating query embedding...")
        try:
            response = ollama.embeddings(
                model=self.embed_model,
                prompt=query
            )
            logger.info("Query embedding generated successfully")
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def vector_search(self, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        logger.info(f"Performing vector search (top_k={k})...")

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Cosine similarity search using pgvector
            query = f"""
                SELECT
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            cur.execute(query, (query_embedding, query_embedding, k))
            results = cur.fetchall()

            cur.close()
            conn.close()

            logger.info(f"Vector search returned {len(results)} results")
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            raise

    def load_corpus_for_bm25(self):
        """Load all documents from database for BM25"""
        logger.info("Loading corpus for BM25...")

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            query = f"""
                SELECT id, content, metadata
                FROM {self.table_name}
                ORDER BY id
            """

            cur.execute(query)
            results = cur.fetchall()

            cur.close()
            conn.close()

            self.corpus_docs = [dict(row) for row in results]

            # Tokenize corpus for BM25
            self.bm25_corpus = [
                word_tokenize(doc['content'].lower())
                for doc in self.corpus_docs
            ]

            # Initialize BM25
            self.bm25_model = BM25Okapi(self.bm25_corpus)

            logger.info(f"Loaded {len(self.corpus_docs)} documents for BM25")

        except Exception as e:
            logger.error(f"Error loading corpus for BM25: {str(e)}")
            raise

    def bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform BM25 search"""
        logger.info(f"Performing BM25 search (top_k={k})...")

        try:
            # Load corpus if not already loaded
            if self.bm25_model is None:
                self.load_corpus_for_bm25()

            # Tokenize query
            tokenized_query = word_tokenize(query.lower())

            # Get BM25 scores
            scores = self.bm25_model.get_scores(tokenized_query)

            # Get top k indices
            top_k_indices = np.argsort(scores)[-k:][::-1]

            # Prepare results
            results = []
            for idx in top_k_indices:
                doc = self.corpus_docs[idx].copy()
                doc['bm25_score'] = float(scores[idx])
                results.append(doc)

            logger.info(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            raise

    def hybrid_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and Vector Search
        Uses reciprocal rank fusion (RRF) to combine results
        """
        if k is None:
            k = self.top_k

        logger.info(f"Starting hybrid search for query: '{query}'")
        logger.info(f"Retrieving top {k} results from each method...")

        try:
            # Generate query embedding
            query_embedding = self.embed_query(query)

            # Perform both searches
            vector_results = self.vector_search(query_embedding, k=k*2)
            bm25_results = self.bm25_search(query, k=k*2)

            # Reciprocal Rank Fusion (RRF)
            # RRF score = sum(1 / (rank + k)) for each ranking
            rrf_k = 60  # Standard RRF constant

            # Calculate RRF scores
            doc_scores = {}

            # Add vector search scores
            for rank, doc in enumerate(vector_results, 1):
                doc_id = doc['id']
                rrf_score = 1.0 / (rank + rrf_k)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'rrf_score': 0,
                        'vector_rank': rank,
                        'bm25_rank': None,
                        'vector_similarity': doc.get('similarity', 0)
                    }
                doc_scores[doc_id]['rrf_score'] += rrf_score

            # Add BM25 scores
            for rank, doc in enumerate(bm25_results, 1):
                doc_id = doc['id']
                rrf_score = 1.0 / (rank + rrf_k)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'rrf_score': 0,
                        'vector_rank': None,
                        'bm25_rank': rank,
                        'bm25_score': doc.get('bm25_score', 0)
                    }
                else:
                    doc_scores[doc_id]['bm25_rank'] = rank
                    doc_scores[doc_id]['bm25_score'] = doc.get('bm25_score', 0)

                doc_scores[doc_id]['rrf_score'] += rrf_score

            # Sort by RRF score
            sorted_docs = sorted(
                doc_scores.values(),
                key=lambda x: x['rrf_score'],
                reverse=True
            )[:k]

            # Format results
            results = []
            for item in sorted_docs:
                doc = item['doc'].copy()
                doc['rrf_score'] = item['rrf_score']
                doc['vector_rank'] = item['vector_rank']
                doc['bm25_rank'] = item['bm25_rank']

                if 'similarity' in item:
                    doc['vector_similarity'] = item['vector_similarity']
                if 'bm25_score' in item:
                    doc['bm25_score'] = item['bm25_score']

                results.append(doc)

            logger.info(f"Hybrid search returned {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using LLM with retrieved context"""
        logger.info("Generating response with LLM...")

        try:
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(context_docs, 1):
                context_parts.append(f"Document {i}:\n{doc['content']}\n")

            context = "\n".join(context_parts)

            # Create prompt
            prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. If you cannot answer the question based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""

            # Generate response using Ollama
            logger.info(f"Calling LLM model: {self.llm_model}...")

            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt
            )

            answer = response['response']

            logger.info("Response generated successfully")
            return answer

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def query(self, user_query: str) -> Dict[str, Any]:
        """Main query method"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing query: {user_query}")
        logger.info(f"{'='*80}")

        try:
            # Perform hybrid search
            retrieved_docs = self.hybrid_search(user_query)

            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return {
                    'query': user_query,
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': []
                }

            # Log retrieved documents
            logger.info(f"\nRetrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                logger.info(f"\n  Document {i}:")
                logger.info(f"    RRF Score: {doc.get('rrf_score', 0):.4f}")
                logger.info(f"    Vector Rank: {doc.get('vector_rank', 'N/A')}")
                logger.info(f"    BM25 Rank: {doc.get('bm25_rank', 'N/A')}")
                logger.info(f"    Source: {doc.get('metadata', {}).get('filename', 'Unknown')}")
                logger.info(f"    Preview: {doc['content'][:150]}...")

            # Generate response
            answer = self.generate_response(user_query, retrieved_docs)

            # Prepare sources
            sources = []
            for doc in retrieved_docs:
                sources.append({
                    'filename': doc.get('metadata', {}).get('filename', 'Unknown'),
                    'source': doc.get('metadata', {}).get('source', 'Unknown'),
                    'rrf_score': doc.get('rrf_score', 0)
                })

            result = {
                'query': user_query,
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources)
            }

            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("Hybrid RAG Query System (BM25 + Vector Search)")
    print("="*80 + "\n")

    try:
        # Initialize RAG system
        rag = HybridRAG()

        # Interactive query loop
        while True:
            print("\n" + "-"*80)
            user_query = input("\nEnter your query (or 'quit' to exit): ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_query:
                print("Please enter a valid query.")
                continue

            try:
                # Process query
                result = rag.query(user_query)

                # Display results
                print("\n" + "="*80)
                print("ANSWER:")
                print("="*80)
                print(result['answer'])

                print("\n" + "="*80)
                print(f"SOURCES ({result['num_sources']} documents):")
                print("="*80)
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n  {i}. {source['filename']}")
                    print(f"     RRF Score: {source['rrf_score']:.4f}")
                    print(f"     Path: {source['source']}")

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\nError: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
