import React, { useState, useEffect } from 'react';
import Chat from './components/Chat';
import Dashboard from './components/Dashboard';
import { metricsAPI } from './services/api';
import './styles/App.css';

function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'dashboard'>('chat');
  const [selectedCompany, setSelectedCompany] = useState<string>('AAPL');
  const [availableCompanies, setAvailableCompanies] = useState<string[]>([]);

  useEffect(() => {
    loadCompanies();
  }, []);

  const loadCompanies = async () => {
    try {
      const response = await metricsAPI.getCompanies();
      setAvailableCompanies(response.companies || []);
      if (response.companies && response.companies.length > 0) {
        setSelectedCompany(response.companies[0]);
      }
    } catch (error) {
      console.error('Error loading companies:', error);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>FinSearch AI</h1>
        <p className="subtitle">Financial Research Co-Pilot</p>
      </header>

      <div className="toolbar">
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            Chat Assistant
          </button>
          <button
            className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            Dashboard
          </button>
        </div>

        <div className="company-selector">
          <label>Company:</label>
          <select
            value={selectedCompany}
            onChange={(e) => setSelectedCompany(e.target.value)}
          >
            {availableCompanies.map((company) => (
              <option key={company} value={company}>
                {company}
              </option>
            ))}
          </select>
        </div>
      </div>

      <main className="app-content">
        {activeTab === 'chat' && <Chat selectedCompany={selectedCompany} />}
        {activeTab === 'dashboard' && <Dashboard company={selectedCompany} />}
      </main>

      <footer className="app-footer">
        <p>Powered by Flan-T5, FinBERT, and RAG</p>
      </footer>
    </div>
  );
}

export default App;
