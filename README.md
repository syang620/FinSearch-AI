# FinSearch AI - Financial Research Co-Pilot

A powerful GenAI-powered research assistant designed for financial analysts and portfolio managers. FinSearch AI helps you digest vast amounts of financial information and answer complex queries about companies, earnings calls, financial statements, and research reports.

## Features

- **RAG-Powered Chat Interface**: Ask questions and get answers enriched with context from your financial documents
- **Sentiment Analysis**: Analyze sentiment from financial documents using FinBERT
- **Interactive Dashboard**: Visualize financial metrics and sentiment trends over time
- **Document Management**: Upload and process PDF, DOCX, Excel, and text files
- **Real-time Metrics**: Display live dashboards with sentiment scores and financial metrics

## Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **LLM**: Flan-T5 or Mistral-7B (via HuggingFace Transformers)
- **RAG**: ChromaDB for vector storage, Sentence Transformers for embeddings
- **Sentiment Analysis**: FinBERT (ProsusAI/finbert)
- **Document Processing**: PyPDF, python-docx, openpyxl, pandas

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Data Visualization**: Recharts
- **API Client**: Axios

## Project Structure

```
FinSearch AI/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # API endpoints
│   │   │   ├── chat.py          # Chat with RAG
│   │   │   ├── documents.py     # Document upload/management
│   │   │   ├── sentiment.py     # Sentiment analysis
│   │   │   └── metrics.py       # Financial metrics
│   │   ├── services/
│   │   │   ├── rag/             # RAG components
│   │   │   │   ├── embeddings.py
│   │   │   │   ├── vector_store.py
│   │   │   │   ├── document_processor.py
│   │   │   │   └── retriever.py
│   │   │   ├── llm/             # LLM service
│   │   │   ├── sentiment/       # Sentiment analysis
│   │   │   └── metrics/         # Metrics service
│   │   ├── core/                # Configuration
│   │   ├── models/              # Pydantic schemas
│   │   └── main.py              # FastAPI app
│   ├── data/
│   │   ├── documents/           # Uploaded documents
│   │   └── chroma_db/           # Vector database
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── Chat.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   ├── MetricsChart.tsx
│   │   │   └── SentimentChart.tsx
│   │   ├── services/api.ts      # API client
│   │   ├── types/               # TypeScript types
│   │   ├── styles/              # CSS files
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional)

### Option 1: Docker Setup (Recommended)

1. Clone the repository:
```bash
cd "FinSearch AI"
```

2. Build and start the services:
```bash
docker-compose up --build
```

3. Access the application:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Option 2: Manual Setup

#### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file:
```bash
cp .env.example .env
# Edit .env to configure settings
```

5. Run the backend:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Run the frontend:
```bash
npm run dev
```

## Usage

### 1. Uploading Documents

Upload financial documents (PDFs, Word docs, Excel files) through the API:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@earnings_call.pdf" \
  -F "company=AAPL" \
  -F "document_type=earnings_call"
```

Or use the API documentation at http://localhost:8000/docs

### 2. Chatting with the Assistant

Use the Chat interface in the frontend to ask questions:
- "What were the key highlights from Apple's latest earnings call?"
- "Compare the revenue growth of AAPL and GOOGL"
- "What is the sentiment around Tesla's recent announcements?"

Toggle "Use RAG" to enable/disable document context in responses.

### 3. Viewing the Dashboard

Select a company from the dropdown and view:
- **Sentiment Analysis**: Time series chart showing sentiment trends
- **Financial Metrics**: Line charts for revenue, EPS, profit margin, etc.

### 4. API Endpoints

#### Chat
- `POST /api/v1/chat/` - Send a message and get AI response
- `GET /api/v1/chat/model-info` - Get LLM model information
- `GET /api/v1/chat/stats` - Get RAG statistics

#### Documents
- `POST /api/v1/documents/upload` - Upload a document
- `GET /api/v1/documents/list` - List all documents
- `DELETE /api/v1/documents/{document_id}` - Delete a document

#### Sentiment
- `POST /api/v1/sentiment/analyze` - Analyze text sentiment
- `GET /api/v1/sentiment/company/{company}` - Get company sentiment history

#### Metrics
- `POST /api/v1/metrics/` - Get financial metrics
- `GET /api/v1/metrics/companies` - List available companies
- `GET /api/v1/metrics/available` - List available metrics

## Configuration

### Backend Configuration (.env)

```env
# LLM Model - Choose one:
LLM_MODEL_NAME=google/flan-t5-base           # Lightweight, faster
# LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1  # More powerful

# Device - Use "cuda" if you have GPU
LLM_DEVICE=cpu

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Sentiment Model
SENTIMENT_MODEL=ProsusAI/finbert
```

### Switching LLM Models

To use Mistral-7B instead of Flan-T5:

1. Edit `backend/.env`:
```env
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
```

2. Restart the backend service

Note: Mistral-7B requires more memory and is slower on CPU. Consider using GPU for better performance.

## Model Information

### Flan-T5
- **Size**: Base (~250M params), Large (~780M params)
- **Best for**: Quick responses, lower resource usage
- **Performance**: Good for straightforward Q&A

### Mistral-7B
- **Size**: ~7B parameters
- **Best for**: Complex reasoning, detailed analysis
- **Performance**: Better quality but slower on CPU

### FinBERT
- **Purpose**: Financial sentiment analysis
- **Trained on**: Financial news and reports
- **Output**: Positive, Negative, or Neutral with confidence scores

## Development

### Running Tests

Backend:
```bash
cd backend
pytest
```

Frontend:
```bash
cd frontend
npm run test
```

### Code Formatting

Backend:
```bash
cd backend
black app/
```

Frontend:
```bash
cd frontend
npm run lint
```

## Performance Considerations

1. **First Run**: Models will be downloaded on first startup (can take several minutes)
2. **CPU vs GPU**: Use GPU for better performance (set `LLM_DEVICE=cuda`)
3. **Memory**: Mistral-7B requires ~14GB RAM, Flan-T5 requires ~2GB RAM
4. **Document Size**: Large documents are chunked automatically

## Troubleshooting

### Backend Issues

**Models not loading:**
- Ensure you have sufficient RAM
- Check internet connection for model downloads
- Models are cached in `~/.cache/huggingface/`

**ChromaDB errors:**
- Delete `backend/data/chroma_db/` and restart

### Frontend Issues

**API connection errors:**
- Ensure backend is running on port 8000
- Check CORS settings in backend config

## Future Enhancements

- [ ] User authentication and multi-tenancy
- [ ] Real-time financial data integration (Yahoo Finance, Alpha Vantage)
- [ ] Support for more document types (HTML, Markdown)
- [ ] Advanced analytics and portfolio tracking
- [ ] Export functionality for reports
- [ ] Mobile responsive design

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- HuggingFace for model hosting and transformers library
- ProsusAI for FinBERT
- ChromaDB for vector database
- FastAPI and React communities
