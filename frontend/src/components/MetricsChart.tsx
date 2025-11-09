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
} from 'recharts';
import { MetricsResponse } from '../types';

interface MetricsChartProps {
  data: MetricsResponse;
}

const COLORS = ['#1976d2', '#f44336', '#4caf50', '#ff9800', '#9c27b0'];

const MetricsChart: React.FC<MetricsChartProps> = ({ data }) => {
  if (!data || !data.metrics) {
    return <div>No metrics data available</div>;
  }

  // Transform data for Recharts
  const metricNames = Object.keys(data.metrics);
  if (metricNames.length === 0) {
    return <div>No metrics selected</div>;
  }

  // Combine all metrics into a single dataset by date
  const chartData: Record<string, any>[] = [];
  const dateMap = new Map<string, Record<string, any>>();

  metricNames.forEach((metricName) => {
    const metricData = data.metrics[metricName];
    metricData.forEach((point) => {
      if (!dateMap.has(point.date)) {
        dateMap.set(point.date, { date: point.date });
      }
      dateMap.get(point.date)![metricName] = point.value;
    });
  });

  chartData.push(...Array.from(dateMap.values()).sort((a, b) =>
    a.date.localeCompare(b.date)
  ));

  return (
    <div style={styles.container}>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          {metricNames.map((metricName, index) => (
            <Line
              key={metricName}
              type="monotone"
              dataKey={metricName}
              stroke={COLORS[index % COLORS.length]}
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          ))}
        </LineChart>
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
};

export default MetricsChart;
