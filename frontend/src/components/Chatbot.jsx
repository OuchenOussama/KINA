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
    weight: null,
    isPregnant: false,
    isBreastfeeding: false,
    allergies: [],
    medicalConditions: [],
    currentMedications: [],
    kidneyFunction: 'normal',
    liverFunction: 'normal',
    heartCondition: false
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
      console.log('Step update received:', data);
      console.log('Current steps before update:', processingSteps);

      setTimeout(() => {
        setProcessingSteps(prev => {
          console.log('Previous steps:', prev);

          const updated = [...prev];
          const completedStepIndex = updated.findIndex(s => s.step === data.step);
          console.log(`Looking for step "${data.step}", found at index:`, completedStepIndex);

          if (completedStepIndex !== -1) {
            updated[completedStepIndex] = {
              ...updated[completedStepIndex],
              status: 'completed',
              result: data.result,
              completedAt: Date.now()
            };

            console.log(`Updated step "${data.step}" to completed`);

            const nextStepIndex = completedStepIndex + 1;
            if (nextStepIndex < updated.length && updated[nextStepIndex].status === 'waiting') {
              updated[nextStepIndex] = {
                ...updated[nextStepIndex],
                status: 'processing',
                startedAt: Date.now()
              };
              console.log(`Set next step "${updated[nextStepIndex].step}" to processing`);
            }
          } else {
            console.log(`Step "${data.step}" not found in current steps!`);
            console.log('Available steps:', prev.map(s => s.step));
          }

          console.log('Final updated steps:', updated);
          return updated;
        });
      }, 0); // Zero delay to push to next event loop tick
    });

    // Listen for final response
    socket.on('final_response', (data) => {
      // Mark all remaining steps as completed
      setProcessingSteps(prev => prev.map(step => ({
        ...step,
        status: step.status === 'processing' || step.status === 'waiting'
          ? 'completed'
          : step.status,
        completedAt: step.status !== 'completed' ? Date.now() : step.completedAt
      })));

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: data.content,
        timestamp: new Date(),
        lang: data.lang,
        metadata: {
          resultsCount: (data.neo4j_results?.length || 0) + (data.hybrid_results?.length || 0),
          processing_steps: data.metadata.processing_steps,
          entities: data.entities,
          neo4j_results: data.neo4j_results,
          hybrid_results: data.hybrid_results
        },
      };

      setMessages(prev => [...prev, botMessage]);

      // Clear processing state after a short delay to show completion
      setTimeout(() => {
        setIsProcessing(false);
        setProcessingSteps([]);
      }, 2000);
    });

    // Listen for errors
    socket.on('error', (data) => {
      // Mark current processing step as error
      setProcessingSteps(prev => prev.map(step => {
        if (step.status === 'processing') {
          return { ...step, status: 'error', errorAt: Date.now() };
        }
        return step;
      }));

      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `Error: ${data.error}${data.details ? ` - ${data.details}` : ''}`,
        timestamp: new Date(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);

      setTimeout(() => {
        setIsProcessing(false);
        setProcessingSteps([]);
      }, 2000);
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

    // Initialize processing steps with proper initial states
    const initialSteps = [
      { step: 'Translation', status: 'processing', startedAt: Date.now() },
      { step: 'Spelling Correction', status: 'waiting' },
      { step: 'NER Extraction', status: 'waiting' },
      { step: 'Cypher Query', status: 'waiting' },
      { step: 'Neo4j Extraction', status: 'waiting' },
      { step: 'Knowledge Retrieval', status: 'waiting' },
      { step: 'Safety Check', status: 'waiting' },
      { step: 'Answer Generation', status: 'waiting' },
    ];

    console.log('Initializing steps:', initialSteps); // Debug log
    setProcessingSteps(initialSteps);

    // Send query to server
    if (socketRef.current) {
      socketRef.current.emit('process_query', {
        query: inputMessage,
        userProfile: userProfile
      });
    }

    setInputMessage('');
    setUserProfile({
      age: null,
      weight: null,
      isPregnant: false,
      isBreastfeeding: false,
      allergies: [],
      medicalConditions: [],
      currentMedications: [],
      kidneyFunction: 'normal',
      liverFunction: 'normal',
      heartCondition: false
    })
  };

  return (
    <div className="h-screen bg-white flex flex-col">
      {/* Header with Profile Toggle */}
      <div className="flex-shrink-0 border-b border-gray-300 p-3 flex justify-between items-center">
        <div className="text-left header-logo">
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