import React, { useState, useEffect } from 'react';
import { metricsAPI, sentimentAPI } from '../services/api';
import { MetricsResponse, CompanySentiment } from '../types';
import MetricsChart from './MetricsChart';
import SentimentChart from './SentimentChart';

interface DashboardProps {
  company: string;
}

const Dashboard: React.FC<DashboardProps> = ({ company }) => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [sentiment, setSentiment] = useState<CompanySentiment | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['revenue', 'eps']);
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadAvailableMetrics();
  }, []);

  useEffect(() => {
    if (company) {
      loadData();
    }
  }, [company, selectedMetrics]);

  const loadAvailableMetrics = async () => {
    try {
      const response = await metricsAPI.getAvailableMetrics();
      setAvailableMetrics(response.metrics || []);
    } catch (error) {
      console.error('Error loading available metrics:', error);
    }
  };

  const loadData = async () => {
    setIsLoading(true);
    try {
      const [metricsData, sentimentData] = await Promise.all([
        metricsAPI.getMetrics(company, selectedMetrics),
        sentimentAPI.getCompanySentiment(company),
      ]);

      setMetrics(metricsData);
      setSentiment(sentimentData);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleMetricToggle = (metric: string) => {
    setSelectedMetrics((prev) => {
      if (prev.includes(metric)) {
        return prev.filter((m) => m !== metric);
      } else {
        return [...prev, metric];
      }
    });
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2>Dashboard: {company}</h2>
        <button onClick={loadData} style={styles.refreshButton}>
          Refresh
        </button>
      </div>

      {isLoading && <div style={styles.loading}>Loading...</div>}

      {!isLoading && (
        <>
          {/* Sentiment Section */}
          <div style={styles.section}>
            <h3>Sentiment Analysis</h3>
            {sentiment && (
              <div>
                <div style={styles.sentimentSummary}>
                  <div style={styles.sentimentBadge}>
                    <span style={styles.sentimentLabel}>Overall Sentiment:</span>
                    <span
                      style={{
                        ...styles.sentimentValue,
                        color: getSentimentColor(sentiment.average_sentiment.label),
                      }}
                    >
                      {sentiment.average_sentiment.label.toUpperCase()}
                    </span>
                    <span style={styles.sentimentScore}>
                      ({(sentiment.average_sentiment.score * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <div style={styles.sentimentBreakdown}>
                    <span style={styles.positive}>
                      Positive: {sentiment.breakdown.positive}
                    </span>
                    <span style={styles.neutral}>
                      Neutral: {sentiment.breakdown.neutral}
                    </span>
                    <span style={styles.negative}>
                      Negative: {sentiment.breakdown.negative}
                    </span>
                  </div>
                </div>
                <SentimentChart data={sentiment.sentiment_data} />
              </div>
            )}
          </div>

          {/* Metrics Section */}
          <div style={styles.section}>
            <h3>Financial Metrics</h3>

            <div style={styles.metricsSelector}>
              <span>Select Metrics:</span>
              <div style={styles.metricsButtons}>
                {availableMetrics.map((metric) => (
                  <button
                    key={metric}
                    onClick={() => handleMetricToggle(metric)}
                    style={{
                      ...styles.metricButton,
                      ...(selectedMetrics.includes(metric)
                        ? styles.metricButtonActive
                        : {}),
                    }}
                  >
                    {metric}
                  </button>
                ))}
              </div>
            </div>

            {metrics && <MetricsChart data={metrics} />}
          </div>
        </>
      )}
    </div>
  );
};

const getSentimentColor = (sentiment: string): string => {
  switch (sentiment) {
    case 'positive':
      return '#4caf50';
    case 'negative':
      return '#f44336';
    default:
      return '#ff9800';
  }
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '20px',
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
  },
  refreshButton: {
    padding: '8px 16px',
    backgroundColor: '#1976d2',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  loading: {
    textAlign: 'center',
    padding: '40px',
    fontSize: '18px',
    color: '#666',
  },
  section: {
    marginBottom: '30px',
  },
  sentimentSummary: {
    marginBottom: '20px',
  },
  sentimentBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '10px',
  },
  sentimentLabel: {
    fontWeight: 'bold',
  },
  sentimentValue: {
    fontSize: '20px',
    fontWeight: 'bold',
  },
  sentimentScore: {
    color: '#666',
  },
  sentimentBreakdown: {
    display: 'flex',
    gap: '20px',
    fontSize: '14px',
  },
  positive: {
    color: '#4caf50',
  },
  neutral: {
    color: '#ff9800',
  },
  negative: {
    color: '#f44336',
  },
  metricsSelector: {
    marginBottom: '20px',
  },
  metricsButtons: {
    display: 'flex',
    gap: '10px',
    marginTop: '10px',
    flexWrap: 'wrap',
  },
  metricButton: {
    padding: '8px 16px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    backgroundColor: 'white',
    cursor: 'pointer',
  },
  metricButtonActive: {
    backgroundColor: '#1976d2',
    color: 'white',
    borderColor: '#1976d2',
  },
};

export default Dashboard;
