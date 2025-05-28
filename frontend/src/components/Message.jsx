// Message.jsx
import React, { useState } from 'react';
import { Bot, User, Database, XCircle, Brain } from 'lucide-react';
import EntitiesModal from './EntitiesModal';

// Simple markdown parser function
const parseMarkdown = (text) => {
  if (!text) return '';
  
  // Convert markdown to HTML
  let html = text
    .replace(/^### (.*$)/gim, '<h3 class="text-base font-semibold mb-2 mt-3">$1</h3>')
    .replace(/^## (.*$)/gim, '<h2 class="text-lg font-semibold mb-2 mt-3">$1</h2>')
    .replace(/^# (.*$)/gim, '<h1 class="text-xl font-bold mb-3 mt-4">$1</h1>')
    .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/__(.*?)__/g, '<strong>$1</strong>')
    .replace(/_(.*?)_/g, '<em>$1</em>')
    .replace(/```([\s\S]*?)```/g, '<pre class="bg-gray-800 text-green-400 p-3 rounded text-xs overflow-x-auto my-2"><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code class="bg-gray-200 px-1 py-0.5 rounded text-xs font-mono">$1</code>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/^\* (.+)$/gim, '<li class="ml-4 mb-1">• $1</li>')
    .replace(/^- (.+)$/gim, '<li class="ml-4 mb-1">• $1</li>')
    .replace(/^\d+\. (.+)$/gim, '<li class="ml-4 mb-1 list-decimal">$1</li>')
    .replace(/\n\n/g, '</p><p class="mb-2">')
    .replace(/\n/g, '<br/>');
  
  if (!html.startsWith('<')) {
    html = `<p class="mb-2">${html}</p>`;
  }
  
  return html;
};

const Message = ({ message }) => {
  const parsedContent = parseMarkdown(message.content);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleEntityClick = () => {
    setIsModalOpen(true);
  };

  const entityCount = message.metadata?.entities
    ? Object.values(message.metadata.entities).reduce((count, value) => {
        if (Array.isArray(value)) {
          return count + value.length;
        } else if (value !== null && value !== undefined) {
          return count + 1;
        }
        return count;
      }, 0)
    : 0;

  return (
    <>
      <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
        <div className={`flex items-start space-x-2 max-w-3xl ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
          <div className={`w-6 h-6 rounded flex items-center justify-center flex-shrink-0 ${message.type === 'user' ? 'bg-black' : 'bg-gray-800'}`}>
            {message.type === 'user' ? (
              <User className="w-3 h-3 text-white" />
            ) : (
              <Bot className="w-3 h-3 text-white" />
            )}
          </div>
          <div className={`px-3 py-2 ${message.type === 'user'
            ? 'bg-black text-white'
            : message.type === 'error'
              ? 'bg-gray-100 text-red-600'
              : 'bg-gray-100 text-black'
            }`}>
            <div
              className="text-sm leading-relaxed"
              dir={message.lang === 'ar' ? 'rtl' : 'ltr'}
              dangerouslySetInnerHTML={{ __html: parsedContent }}
            />

            {message.metadata && (
              <div className="mt-2 pt-2 border-t border-white/20 text-xs flex items-center space-x-3">
                <span className="flex items-center space-x-1">
                  <Database className="w-3 h-3" />
                  <span>{message.metadata.resultsCount || 0}</span>
                </span>
                <span className="flex items-center space-x-1 cursor-pointer hover:underline" onClick={handleEntityClick}>
                  <Brain className="w-3 h-3" />
                  <span>{entityCount}</span>
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
      <EntitiesModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        entities={message.metadata?.entities || {}}
      />
    </>
  );
};

export default Message;