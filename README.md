# Hybrid RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that combines BM25 and Vector Search for optimal document retrieval and question answering.

## Features

- **Multi-format Document Support**: PDF, TXT, DOCX, XLSX
- **Advanced PDF Processing**: Uses Docling library for accurate PDF text extraction
- **Text Cleaning**: NLTK-based text preprocessing and cleaning
- **Intelligent Chunking**: LangChain RecursiveCharacterTextSplitter with configurable chunk sizes
- **Local Embeddings**: Ollama with nomic-embed-text model (no API keys required)
- **Hybrid Search**: Combines BM25 (keyword-based) and Vector Search (semantic) using Reciprocal Rank Fusion (RRF)
- **Vector Database**: PostgreSQL with pgvector extension
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Error Handling**: Robust error handling throughout the pipeline

## Architecture

```
┌─────────────────┐
│  Document Input │
│ (PDF/TXT/DOCX/  │
│     XLSX)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Docling (PDF) │
│  File Readers   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NLTK Cleaning  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LangChain      │
│  Text Splitter  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Ollama Embed   │
│  (nomic-embed)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PostgreSQL +   │
│    pgvector     │
└────────┬────────┘
         │
         ▼
    ┌────────────┐
    │   Query    │
    └─────┬──────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌────────┐
│  BM25  │  │ Vector │
│ Search │  │ Search │
└───┬────┘  └───┬────┘
    │           │
    └─────┬─────┘
          │
          ▼
    ┌──────────┐
    │   RRF    │
    │ Fusion   │
    └─────┬────┘
          │
          ▼
    ┌──────────┐
    │   LLM    │
    │ Response │
    └──────────┘
```

## Prerequisites

- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- Ollama installed locally
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hybridRAG.git
cd hybridRAG
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PostgreSQL and pgvector

#### Windows:
1. Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Install pgvector:
   ```bash
   # Download from https://github.com/pgvector/pgvector/releases
   # Or use pre-built binaries
   ```

#### Linux:
```bash
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-server-dev-all
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### Mac:
```bash
brew install postgresql
brew install pgvector
```

### 5. Install Ollama and Download Models

#### Windows/Mac:
Download from [ollama.ai](https://ollama.ai)

#### Linux:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Pull Required Models:
```bash
ollama pull nomic-embed-text:latest
ollama pull gpt-OSS:20b-cloud  # Or your preferred LLM model
```

### 6. Configure Environment Variables

Edit the `.env` file with your settings:

```env
# PostgreSQL Configuration
POSTGRES_DB=product_costing
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Table Names
VECTOR_TABLE_NAME=sap_controlling

# Embedding Model (Ollama)
OLLAMA_EMBED_MODEL=nomic-embed-text:latest
OLLAMA_BASE_URL=http://localhost:11434

# LLM Model
LLM_MODEL_NAME=gpt-OSS:20b-cloud
LLM_PROVIDER=ollama

# RAG Settings
CHUNK_SIZE=1024
CHUNK_OVERLAP=256
TOP_K=7
```

## Usage

### Step 1: Ingest Documents

Place your documents (PDF, TXT, DOCX, XLSX) in a folder and run:

```bash
python ingest.py <path_to_your_documents_folder>
```

Example:
```bash
python ingest.py ./data
```

The ingestion pipeline will:
1. Read all supported files from the folder
2. Convert PDFs using Docling
3. Clean text using NLTK
4. Chunk documents using LangChain
5. Generate embeddings using Ollama
6. Store in PostgreSQL with pgvector

**Logs**: Check `ingest_YYYYMMDD_HHMMSS.log` for detailed progress.

### Step 2: Query the System

Start the interactive query interface:

```bash
python query.py
```

Then enter your questions:

```
Enter your query (or 'quit' to exit): What is the pricing strategy?
```

The system will:
1. Convert your query to embeddings
2. Perform BM25 keyword search
3. Perform vector similarity search
4. Combine results using Reciprocal Rank Fusion (RRF)
5. Generate answer using the LLM with retrieved context

**Logs**: Check `query_YYYYMMDD_HHMMSS.log` for detailed information.

## Hybrid Search Explanation

This system uses **Reciprocal Rank Fusion (RRF)** to combine two complementary search methods:

### BM25 (Keyword Search)
- **Strengths**: Exact keyword matching, works well for specific terms, acronyms, and entity names
- **Weakness**: Doesn't understand semantic meaning

### Vector Search (Semantic Search)
- **Strengths**: Understands meaning and context, finds semantically similar content
- **Weakness**: May miss exact keyword matches

### RRF Fusion
Combines both methods for superior results:
- Each method ranks documents independently
- RRF score = Σ(1 / (rank + 60)) across both rankings
- Documents appearing in both result sets get higher scores
- Final ranking balances keyword precision and semantic understanding

## Project Structure

```
hybridRAG/
├── ingest.py              # Data ingestion pipeline
├── query.py               # Hybrid RAG query system
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── README.md             # This file
├── .gitignore            # Git ignore file
└── logs/                 # Log files (auto-generated)
    ├── ingest_*.log
    └── query_*.log
```

## Configuration Options

### Chunking Parameters

Adjust in `.env`:
- `CHUNK_SIZE`: Size of text chunks (default: 1024)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 256)

Smaller chunks = more precise but less context
Larger chunks = more context but less precise

### Retrieval Parameters

- `TOP_K`: Number of documents to retrieve (default: 7)

Higher values = more context but slower and potentially noisy
Lower values = faster but might miss relevant information

## Troubleshooting

### Issue: PostgreSQL Connection Error
**Solution**:
- Check PostgreSQL is running: `pg_isready`
- Verify connection details in `.env`
- Ensure database exists: `createdb product_costing`

### Issue: Ollama Connection Error
**Solution**:
- Check Ollama is running: `ollama list`
- Verify Ollama is on port 11434
- Pull models: `ollama pull nomic-embed-text:latest`

### Issue: PDF Extraction Errors
**Solution**:
- Ensure Docling is properly installed: `pip install --upgrade docling`
- Check PDF is not corrupted
- Try with a different PDF

### Issue: Out of Memory
**Solution**:
- Reduce `CHUNK_SIZE` in `.env`
- Process files in smaller batches
- Reduce `TOP_K` for queries

### Issue: Slow Query Performance
**Solution**:
- Ensure vector index is created (happens automatically)
- Reduce `TOP_K` value
- Use faster LLM model

## Performance Tips

1. **Batch Processing**: Process multiple documents at once for efficiency
2. **Index Optimization**: PostgreSQL creates IVFFlat index automatically for fast searches
3. **Caching**: BM25 corpus is cached in memory for faster subsequent queries
4. **Model Selection**: Use smaller LLM models for faster responses

## Advanced Usage

### Custom LLM Models

Edit `.env` to use different models:

```env
# Use different Ollama model
LLM_MODEL_NAME=llama2
LLM_PROVIDER=ollama

# Or use OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Database Maintenance

```sql
-- Check table size
SELECT pg_size_pretty(pg_total_relation_size('sap_controlling'));

-- Count documents
SELECT COUNT(*) FROM sap_controlling;

-- Clear all data (careful!)
TRUNCATE TABLE sap_controlling;
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [Docling](https://github.com/DS4SD/docling) - PDF conversion
- [LangChain](https://www.langchain.com/) - Text processing
- [Ollama](https://ollama.ai/) - Local LLM inference
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - BM25 implementation

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Roadmap

- [ ] Add support for more document types (HTML, MD, CSV)
- [ ] Implement query result caching
- [ ] Add web interface (Gradio/Streamlit)
- [ ] Multi-language support
- [ ] Query history and analytics
- [ ] Document metadata filtering
- [ ] Batch query processing
- [ ] API endpoint support

---

**Built with ❤️ for efficient and accurate document retrieval**
