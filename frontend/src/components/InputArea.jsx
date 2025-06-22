import React, { useState } from 'react';
import { Send, Globe, Database, FileText, Shield, HelpCircle } from 'lucide-react';

const InputArea = ({ inputMessage, setInputMessage, handleSendMessage, isProcessing, setK }) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const [currentK, setCurrentK] = useState(5);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleKChange = (e) => {
    const newK = parseInt(e.target.value);
    setCurrentK(newK);
    setK(newK);
  };

  const getPrecisionRecallLabel = (k) => {
    if (k <= 8) return { text: "High precision, Low recall", color: "text-green-600" };
    if (k <= 15) return { text: "Moderate precision, Moderate recall", color: "text-yellow-600" };
    return { text: "Low precision, High recall", color: "text-red-600" };
  };

  const precisionRecall = getPrecisionRecallLabel(currentK);

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
        <div className="flex items-center space-x-6">
          {/* Stylish drug count selector */}
          <div className="flex items-center space-x-3 bg-gray-50 px-3 py-2 rounded-lg border">
            <label htmlFor="drug-count" className="text-gray-700 font-medium">
              Results:
            </label>
            <select
              id="drug-count"
              value={currentK}
              onChange={handleKChange}
              className="bg-transparent text-gray-900 font-medium focus:outline-none cursor-pointer"
            >
              {Array.from({ length: 21 }, (_, i) => i + 5).map(num => (
                <option key={num} value={num}>{num}</option>
              ))}
            </select>
          </div>
          
          {/* Precision/Completeness indicator */}
          <div className="flex items-center space-x-2">
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${
              currentK <= 8 ? 'bg-green-100 text-green-700' :
              currentK <= 15 ? 'bg-yellow-100 text-yellow-700' :
              'bg-red-100 text-red-700'
            }`}>
              {getPrecisionRecallLabel(currentK).text.replace('recall', 'completeness')}
            </div>
            <div className="relative">
              <HelpCircle 
                className="w-4 h-4 text-gray-400 cursor-help hover:text-gray-600 transition-colors"
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
              />
              {showTooltip && (
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg shadow-xl whitespace-nowrap z-10">
                  <div className="text-center">
                    <div>More results = higher completeness but</div>
                    <div>may include less relevant matches</div>
                  </div>
                  <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                </div>
              )}
            </div>
          </div>
        </div>
        <span>v1.0.0</span>
      </div>
    </div>
  );
};

export default InputArea;