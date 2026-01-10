# Tree-Based RAG (No Vector DB)

A hierarchical Retrieval-Augmented Generation (RAG) system that uses tree structures instead of vector databases for document search and retrieval.

## Overview

This project implements a novel approach to RAG by:
- Converting PDFs to structured markdown using hybrid OCR
- Building hierarchical document trees from markdown structure
- Using Reciprocal Rank Fusion (RRF) with weighted semantic/lexical scoring
- Performing tree-based search without vector databases

## Features

- **Hybrid OCR Processing**: Fast PyMuPDF text extraction + Mistral AI vision model for scanned pages
- **AI-Powered Structuring**: Automatic markdown structure generation for unstructured documents
- **Weighted RRF Search**: Combines BM25 (lexical) and BERT (semantic) with configurable weights
- **Softmax Probability Scores**: Search results shown as interpretable probabilities
- **Interactive UI**: Streamlit-based interface with search path visualization and document tree explorer
- **Efficient Processing**: Only processes new documents, caches results in session state

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd no-vector-RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Mistral API key:
```bash
cp .env.example .env
```

Then edit `.env` and add:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload a PDF document or load the sample manual

3. Ask questions about your document using natural language

4. View search results with:
   - Relevant section content
   - Search path visualization showing algorithm decisions
   - Complete document hierarchy with probability scores

## Architecture

### Core Components

**`app.py`**: Streamlit UI and application logic
- Session state management
- PDF upload and processing workflow
- Search interface and results display

**`core_logic.py`**: Core RAG algorithms
- `DocNode`: Tree node class with RRF scoring
- `parse_markdown_to_tree()`: Builds hierarchical tree from markdown
- `tree_search()`: Traverses tree using weighted RRF scores
- `convert_to_markdown_with_progress()`: Hybrid OCR with Mistral AI
- `get_embedding_model()`: Cached sentence transformer loading

### Search Algorithm

1. **Document Processing**:
   - Extract text with PyMuPDF (fast)
   - Use Mistral Pixtral OCR only for scanned pages
   - Structure content with Mistral Large if needed
   - Parse markdown into hierarchical tree

2. **Tree Search**:
   - At each tree level, score all children using RRF
   - RRF combines BM25 (lexical) and BERT (semantic) scores
   - Semantic weighted 2x higher than lexical
   - Apply softmax to convert scores to probabilities
   - Navigate to highest-probability child
   - Repeat until leaf node or depth limit

3. **Scoring Formula**:
   ```python
   # Weighted RRF
   score = 1.0 * (1/(k + bm25_rank)) + 2.0 * (1/(k + bert_rank))

   # Softmax conversion to probability
   P(i) = exp(score_i) / Σ exp(score_j)
   ```

## Configuration

### Model Settings

- **Embedding Model**: `all-MiniLM-L6-v2` (cached locally in `./local_all_minilm/`)
- **OCR Model**: Mistral Pixtral 12B
- **Structuring Model**: Mistral Large

### Search Parameters

Configurable in `core_logic.py`:
- `k=20`: RRF rank constant
- `semantic_weight=2.0`: Weight for BERT semantic scores
- `lexical_weight=1.0`: Weight for BM25 lexical scores
- `depth_limit=5`: Maximum tree traversal depth

## Dependencies

### Core
- `torch`, `torchvision`, `torchaudio`: PyTorch (CPU version)

### OCR & Document Processing
- `mistralai`: Mistral AI API client
- `PyMuPDF`: Fast PDF text extraction
- `Pillow`: Image processing

### RAG & Search
- `sentence-transformers`: BERT embeddings
- `rank_bm25`: BM25 lexical scoring
- `streamlit`: Web UI
- `numpy`: Numerical operations

## Project Structure

```
no-vector-RAG/
├── app.py                      # Streamlit application
├── core_logic.py               # RAG algorithms and tree search
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not in git)
├── .env.example               # Template for .env
├── knowledge base/            # Sample documents
│   └── manual_for_pumps.md
└── local_all_minilm/          # Cached embedding model
```

## How It Differs from Vector RAG

**Traditional Vector RAG**:
- Chunks documents into fixed-size segments
- Generates embeddings for all chunks
- Stores in vector database (ChromaDB, Pinecone, etc.)
- Performs similarity search across all chunks

**Tree-Based RAG (This Project)**:
- Preserves document hierarchy and structure
- Navigates tree level-by-level using hybrid scoring
- No vector database needed
- Maintains context through tree relationships
- Better for structured documents with clear hierarchies

## Performance

- **PDF Conversion**: Varies by document (fast text extraction, slower AI OCR for scanned pages)
- **Tree Building**: < 1 second for most documents
- **Search**: Near-instant (no database queries)
- **Model Loading**: One-time download, then cached locally

## Contributing

Contributions welcome! Areas for improvement:
- Support for more document formats
- Alternative embedding models
- Configurable scoring weights via UI
- Export search results
- Multi-document search

## License

See LICENSE file for details.

## Acknowledgments

- Mistral AI for OCR and structuring models
- Sentence Transformers for embedding models
- Streamlit for the UI framework
