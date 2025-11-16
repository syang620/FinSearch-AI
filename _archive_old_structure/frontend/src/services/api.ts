import axios from 'axios';
import {
  ChatRequest,
  ChatResponse,
  CompanySentiment,
  MetricsResponse,
  DocumentResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Chat API
export const chatAPI = {
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>('/chat/', request);
    return response.data;
  },

  getModelInfo: async () => {
    const response = await apiClient.get('/chat/model-info');
    return response.data;
  },

  getStats: async () => {
    const response = await apiClient.get('/chat/stats');
    return response.data;
  },
};

// Documents API
export const documentsAPI = {
  upload: async (
    file: File,
    company?: string,
    documentType?: string
  ): Promise<DocumentResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (company) formData.append('company', company);
    if (documentType) formData.append('document_type', documentType);

    const response = await apiClient.post<DocumentResponse>(
      '/documents/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  list: async () => {
    const response = await apiClient.get('/documents/list');
    return response.data;
  },

  delete: async (documentId: string) => {
    const response = await apiClient.delete(`/documents/${documentId}`);
    return response.data;
  },
};

// Sentiment API
export const sentimentAPI = {
  analyze: async (text: string, company?: string) => {
    const response = await apiClient.post('/sentiment/analyze', {
      text,
      company,
    });
    return response.data;
  },

  getCompanySentiment: async (company: string): Promise<CompanySentiment> => {
    const response = await apiClient.get<CompanySentiment>(
      `/sentiment/company/${company}`
    );
    return response.data;
  },
};

// Metrics API
export const metricsAPI = {
  getMetrics: async (
    company: string,
    metricNames: string[],
    startDate?: string,
    endDate?: string
  ): Promise<MetricsResponse> => {
    const response = await apiClient.post<MetricsResponse>('/metrics/', {
      company,
      metric_names: metricNames,
      start_date: startDate,
      end_date: endDate,
    });
    return response.data;
  },

  getCompanies: async () => {
    const response = await apiClient.get('/metrics/companies');
    return response.data;
  },

  getAvailableMetrics: async () => {
    const response = await apiClient.get('/metrics/available');
    return response.data;
  },

  getLatestMetric: async (company: string, metricName: string) => {
    const response = await apiClient.get(`/metrics/${company}/${metricName}`);
    return response.data;
  },
};

export default {
  chat: chatAPI,
  documents: documentsAPI,
  sentiment: sentimentAPI,
  metrics: metricsAPI,
};
