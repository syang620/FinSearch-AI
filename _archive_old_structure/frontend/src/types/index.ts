// Chat types
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface ChatRequest {
  query: string;
  conversation_id?: string;
  use_rag: boolean;
  company_filter?: string;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  sources?: Source[];
  timestamp: string;
}

export interface Source {
  text: string;
  metadata: Record<string, any>;
  relevance_score?: number;
}

// Sentiment types
export interface SentimentScore {
  label: 'positive' | 'negative' | 'neutral';
  score: number;
}

export interface SentimentData {
  date: string;
  label: string;
  score: number;
}

export interface CompanySentiment {
  company: string;
  sentiment_data: SentimentData[];
  average_sentiment: SentimentScore;
  breakdown: {
    positive: number;
    negative: number;
    neutral: number;
  };
}

// Metrics types
export interface MetricDataPoint {
  date: string;
  value: number;
  metric_name: string;
}

export interface MetricsResponse {
  company: string;
  metrics: Record<string, MetricDataPoint[]>;
  metadata?: Record<string, any>;
}

// Document types
export interface DocumentUpload {
  file: File;
  company?: string;
  document_type?: string;
}

export interface DocumentResponse {
  document_id: string;
  filename: string;
  status: string;
  chunks_created: number;
  message: string;
}
