from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Chat models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query")
    conversation_id: Optional[str] = None
    use_rag: bool = Field(True, description="Whether to use RAG for context")
    company_filter: Optional[str] = Field(None, description="Filter by specific company")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant response")
    conversation_id: str
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents used")
    timestamp: datetime


# Document models
class DocumentUpload(BaseModel):
    filename: str
    content_type: str
    company: Optional[str] = None
    document_type: Optional[str] = Field(None, description="e.g., '10-K', 'earnings_call', 'research_report'")


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_created: int
    message: str


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    company: Optional[str]
    document_type: Optional[str]
    upload_date: datetime
    chunk_count: int


# Sentiment models
class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    company: Optional[str] = None


class SentimentScore(BaseModel):
    label: str = Field(..., description="positive, negative, or neutral")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class SentimentResponse(BaseModel):
    sentiment: SentimentScore
    text_snippet: str = Field(..., description="First 200 chars of analyzed text")
    timestamp: datetime


class CompanySentimentHistory(BaseModel):
    company: str
    sentiment_data: List[Dict[str, Any]] = Field(..., description="Time series sentiment data")
    average_sentiment: SentimentScore


# Metrics models
class MetricRequest(BaseModel):
    company: str
    metric_names: List[str] = Field(..., description="e.g., ['revenue', 'eps', 'profit_margin']")
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class MetricDataPoint(BaseModel):
    date: str
    value: float
    metric_name: str


class MetricsResponse(BaseModel):
    company: str
    metrics: Dict[str, List[MetricDataPoint]]
    metadata: Optional[Dict[str, Any]] = None


# General response
class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
