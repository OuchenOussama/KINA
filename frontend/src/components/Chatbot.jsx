import React, { useState, useRef, useEffect } from 'react';
import { Bot, Shield, Zap } from 'lucide-react';
import { io } from 'socket.io-client';
import Message from './Message';
import InputArea from './InputArea';
import ProfilePanel from './ProfilePanel';
import ProcessingSteps from './ProcessingSteps';

const PharmaceuticalChatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m your pharmaceutical AI assistant powered by advanced knowledge graphs and hybrid retrieval systems. I can help you with drug information, interactions, side effects, and safety guidelines. How can I assist you today?',
      timestamp: new Date(),
    },
  ]);

  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingSteps, setProcessingSteps] = useState([]);
  const [userProfile, setUserProfile] = useState({
    age: null,
    isPregnant: false,
    allergies: [],
  });
  const [showProfile, setShowProfile] = useState(false);
  const messagesEndRef = useRef(null);
  const socketRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initialize socket connection
    socketRef.current = io('http://localhost:5000'); // Replace with your server URL

    const socket = socketRef.current;

    // Listen for step updates
    socket.on('step_update', (data) => {
      console.log('Step update received:', data); // Debug log
      setProcessingSteps(prev => {
        const stepIndex = prev.findIndex(s => s.step === data.step);
        if (stepIndex === -1) return prev;

        const updated = prev.map((s, index) => {
          if (s.step === data.step) {
            // Update current step to completed
            return { ...s, status: 'completed', result: data.result };
          } else if (index === stepIndex + 1 && s.status === 'waiting') {
            // Set next step to processing
            return { ...s, status: 'processing' };
          }
          return s;
        });

        console.log('Updated steps:', updated); // Debug log
        return updated;
      });
    });

    // Listen for final response
    socket.on('final_response', (data) => {
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: data.content,
        timestamp: new Date(),
        metadata: {
          resultsCount: (data.neo4j_results?.length || 0) + (data.hybrid_results?.length || 0),
          processing_steps: data.metadata.processing_steps,
          entities: data.entities,
          neo4j_results: data.neo4j_results,
          hybrid_results: data.hybrid_results
        },
      };
      setMessages(prev => [...prev, botMessage]);
      setTimeout(() => {
        setIsProcessing(false);
        setProcessingSteps([]);
      }, 1000);
    });

    // Listen for errors
    socket.on('error', (data) => {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `Error: ${data.error}${data.details ? ` - ${data.details}` : ''}`,
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsProcessing(false);
      setProcessingSteps([]);
    });

    // Cleanup on unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  const handleSendMessage = () => {
    if (!inputMessage.trim() || isProcessing) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsProcessing(true);

    // Initialize processing steps
    setProcessingSteps([
      { step: 'Translation', status: 'processing', timestamp: Date.now() },
      { step: 'Spelling Correction', status: 'processing', timestamp: Date.now() },
      { step: 'NER Extraction', status: 'processing', timestamp: Date.now() },
      { step: 'Cypher Query', status: 'processing', timestamp: Date.now() },
      { step: 'Neo4j Extraction', status: 'processing', timestamp: Date.now() },
      { step: 'Query Template Conversion', status: 'processing', timestamp: Date.now() },
      { step: 'Knowledge Retrieval', status: 'processing', timestamp: Date.now() },
      { step: 'Safety Check', status: 'processing', timestamp: Date.now() },
      { step: 'Answer Generation', status: 'processing', timestamp: Date.now() },
    ]);

    // Send query to server
    if (socketRef.current) {
      socketRef.current.emit('process_query', {
        query: inputMessage,
        userProfile: userProfile
      });
    }

    setInputMessage('');
  };


  return (
    <div className="h-screen bg-white flex flex-col">
      {/* Header with Profile Toggle */}
      <div className="flex-shrink-0 border-b border-gray-300 p-3 flex justify-between items-center">
        <div className="text-left">
          <h1 className="text-lg font-bold text-black tracking-wide">KINA</h1>
          <p className="text-xs text-gray-600 mt-1">
            <span className="font-semibold">K</span>nowledge Graphs{' '}
            <span className="font-semibold">I</span>ntegrated with{' '}
            <span className="font-semibold">N</span>LP for{' '}
            <span className="font-semibold">A</span>pothecary
          </p>
        </div>
        <button
          onClick={() => setShowProfile(!showProfile)}
          className="p-2 bg-black text-white hover:bg-gray-800"
          title="Safety Profile"
        >
          <Shield className="w-4 h-4" />
        </button>
      </div>

      {/* Profile Panel */}
      {showProfile && (
        <div className="flex-shrink-0">
          <ProfilePanel userProfile={userProfile} setUserProfile={setUserProfile} />
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-5xl mx-auto">
          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}
          {isProcessing && <ProcessingSteps steps={processingSteps} />}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0">
        <InputArea
          inputMessage={inputMessage}
          setInputMessage={setInputMessage}
          handleSendMessage={handleSendMessage}
          isProcessing={isProcessing}
        />
      </div>
    </div>
  );
};

export default PharmaceuticalChatbot;
