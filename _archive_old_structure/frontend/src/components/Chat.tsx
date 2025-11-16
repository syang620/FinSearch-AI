import React, { useState, useRef, useEffect } from 'react';
import { chatAPI } from '../services/api';
import { ChatMessage, ChatResponse } from '../types';

interface ChatProps {
  selectedCompany?: string;
}

const Chat: React.FC<ChatProps> = ({ selectedCompany }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useRAG, setUseRAG] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response: ChatResponse = await chatAPI.sendMessage({
        query: input,
        use_rag: useRAG,
        company_filter: selectedCompany,
      });

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        timestamp: response.timestamp,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2>Financial Research Assistant</h2>
        <div style={styles.controls}>
          <label style={styles.checkbox}>
            <input
              type="checkbox"
              checked={useRAG}
              onChange={(e) => setUseRAG(e.target.checked)}
            />
            Use RAG (Document Context)
          </label>
          {selectedCompany && (
            <span style={styles.companyBadge}>Company: {selectedCompany}</span>
          )}
        </div>
      </div>

      <div style={styles.messagesContainer}>
        {messages.length === 0 && (
          <div style={styles.emptyState}>
            <p>Ask me anything about financial research!</p>
            <p style={styles.hint}>
              Try: "What are the key metrics for evaluating a tech company?"
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              ...styles.message,
              ...(message.role === 'user' ? styles.userMessage : styles.assistantMessage),
            }}
          >
            <div style={styles.messageRole}>
              {message.role === 'user' ? 'You' : 'Assistant'}
            </div>
            <div style={styles.messageContent}>{message.content}</div>
          </div>
        ))}

        {isLoading && (
          <div style={styles.loading}>
            <div style={styles.loadingDot}></div>
            <div style={styles.loadingDot}></div>
            <div style={styles.loadingDot}></div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div style={styles.inputContainer}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your question..."
          style={styles.input}
          rows={3}
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          style={{
            ...styles.sendButton,
            ...(isLoading || !input.trim() ? styles.sendButtonDisabled : {}),
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  },
  header: {
    padding: '20px',
    borderBottom: '1px solid #e0e0e0',
  },
  controls: {
    marginTop: '10px',
    display: 'flex',
    alignItems: 'center',
    gap: '15px',
  },
  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    cursor: 'pointer',
  },
  companyBadge: {
    padding: '5px 10px',
    backgroundColor: '#e3f2fd',
    borderRadius: '4px',
    fontSize: '14px',
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
  },
  emptyState: {
    textAlign: 'center',
    color: '#666',
    marginTop: '50px',
  },
  hint: {
    fontSize: '14px',
    color: '#999',
    marginTop: '10px',
  },
  message: {
    marginBottom: '15px',
    padding: '12px',
    borderRadius: '8px',
    maxWidth: '80%',
  },
  userMessage: {
    marginLeft: 'auto',
    backgroundColor: '#e3f2fd',
  },
  assistantMessage: {
    marginRight: 'auto',
    backgroundColor: '#f5f5f5',
  },
  messageRole: {
    fontWeight: 'bold',
    fontSize: '12px',
    marginBottom: '5px',
    color: '#666',
  },
  messageContent: {
    lineHeight: '1.5',
  },
  loading: {
    display: 'flex',
    gap: '5px',
    padding: '20px',
  },
  loadingDot: {
    width: '8px',
    height: '8px',
    backgroundColor: '#666',
    borderRadius: '50%',
    animation: 'pulse 1.5s infinite',
  },
  inputContainer: {
    padding: '20px',
    borderTop: '1px solid #e0e0e0',
    display: 'flex',
    gap: '10px',
  },
  input: {
    flex: 1,
    padding: '12px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '14px',
    fontFamily: 'inherit',
    resize: 'none',
  },
  sendButton: {
    padding: '12px 24px',
    backgroundColor: '#1976d2',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: 'bold',
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed',
  },
};

export default Chat;
