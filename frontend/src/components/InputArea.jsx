import React from 'react';
import { Send, Globe, Database, FileText, Shield } from 'lucide-react';

const InputArea = ({ inputMessage, setInputMessage, handleSendMessage, isProcessing }) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="border-t border-gray-300 bg-white p-4">
      <div className="flex items-end space-x-3">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about medications..."
          className="flex-1 resize-none border border-gray-300 px-3 py-2 text-sm text-black placeholder-gray-500 focus:outline-none focus:border-black"
          rows={1}
          style={{ minHeight: '36px', maxHeight: '100px' }}
          disabled={isProcessing}
        />
        <button
          onClick={handleSendMessage}
          disabled={!inputMessage.trim() || isProcessing}
          className="px-4 py-2 bg-black text-white text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-800"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
      <div className="flex items-center justify-between mt-3 text-xs text-gray-600">
        <div className="flex items-center space-x-4">
          <span className="flex items-center space-x-1">
            <Globe className="w-3 h-3" />
            <span>Multi-language</span>
          </span>
          <span className="flex items-center space-x-1">
            <Database className="w-3 h-3" />
            <span>Neo4j</span>
          </span>
          <span className="flex items-center space-x-1">
            <FileText className="w-3 h-3" />
            <span>Hybrid KG</span>
          </span>
          <span className="flex items-center space-x-1">
            <Shield className="w-3 h-3" />
            <span>Safety</span>
          </span>
        </div>
        <span>v1.0.0</span>
      </div>
    </div>
  );
};


export default InputArea;