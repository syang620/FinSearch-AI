import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { SentimentData } from '../types';

interface SentimentChartProps {
  data: SentimentData[];
}

const SentimentChart: React.FC<SentimentChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return <div>No sentiment data available</div>;
  }

  // Transform sentiment labels to numeric values for charting
  const chartData = data.map((item) => ({
    date: item.date,
    score: item.score,
    sentiment: item.label,
    // Convert sentiment to numeric: positive=1, neutral=0, negative=-1
    sentimentValue:
      item.label === 'positive' ? 1 : item.label === 'negative' ? -1 : 0,
    // Weighted score
    weightedScore:
      (item.label === 'positive' ? 1 : item.label === 'negative' ? -1 : 0) *
      item.score,
  }));

  return (
    <div style={styles.container}>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="colorSentiment" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#4caf50" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#4caf50" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            domain={[-1, 1]}
            ticks={[-1, -0.5, 0, 0.5, 1]}
            tickFormatter={(value) => {
              if (value === 1) return 'Positive';
              if (value === -1) return 'Negative';
              if (value === 0) return 'Neutral';
              return '';
            }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                return (
                  <div style={styles.tooltip}>
                    <p style={styles.tooltipDate}>{data.date}</p>
                    <p
                      style={{
                        ...styles.tooltipSentiment,
                        color:
                          data.sentiment === 'positive'
                            ? '#4caf50'
                            : data.sentiment === 'negative'
                            ? '#f44336'
                            : '#ff9800',
                      }}
                    >
                      {data.sentiment.toUpperCase()}
                    </p>
                    <p style={styles.tooltipScore}>
                      Score: {(data.score * 100).toFixed(1)}%
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Area
            type="monotone"
            dataKey="weightedScore"
            stroke="#1976d2"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorSentiment)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '100%',
    padding: '20px',
    backgroundColor: '#f9f9f9',
    borderRadius: '8px',
  },
  tooltip: {
    backgroundColor: 'white',
    padding: '10px',
    border: '1px solid #ddd',
    borderRadius: '4px',
  },
  tooltipDate: {
    margin: 0,
    fontSize: '12px',
    color: '#666',
  },
  tooltipSentiment: {
    margin: '5px 0',
    fontSize: '14px',
    fontWeight: 'bold',
  },
  tooltipScore: {
    margin: 0,
    fontSize: '12px',
    color: '#666',
  },
};

export default SentimentChart;
