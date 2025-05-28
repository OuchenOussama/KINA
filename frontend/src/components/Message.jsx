import React from 'react';
import { Bot, User, Database, XCircle, Brain } from 'lucide-react';


const Message = ({ message }) => {
  return (
    <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`flex items-start space-x-2 max-w-3xl ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
        <div className={`w-6 h-6 rounded flex items-center justify-center flex-shrink-0 ${
          message.type === 'user' ? 'bg-black' : 'bg-gray-800'
        }`}>
          {message.type === 'user' ? (
            <User className="w-3 h-3 text-white" />
          ) : (
            <Bot className="w-3 h-3 text-white" />
          )}
        </div>
        <div className={`px-3 py-2 ${
          message.type === 'user'
            ? 'bg-black text-white'
            : message.type === 'error'
            ? 'bg-gray-100 text-red-600'
            : 'bg-gray-100 text-black'
        }`}>
          <div className="text-sm leading-relaxed">{message.content}</div>
          {message.metadata && (
            <div className="mt-2 pt-2 border-t border-white/20 text-xs flex items-center space-x-3">
              <span className="flex items-center space-x-1">
                <Database className="w-3 h-3" />
                <span>{message.metadata.resultsCount || 0}</span>
              </span>
              <span className="flex items-center space-x-1">
                <Brain className="w-3 h-3" />
                <span>{message.metadata.processing_steps?.find(step => step.step === 'NER Extraction')?.result?.entities?.length || 0}</span>
              </span>
            </div>
          )}
          {message.type === 'error' && (
            <div className="mt-1 text-xs flex items-center space-x-1">
              <XCircle className="w-3 h-3" />
              <span>Error occurred</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;