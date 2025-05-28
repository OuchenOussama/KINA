import React from 'react';
import { Loader2, CheckCircle, XCircle, Sparkles, Clock } from 'lucide-react';

const ProcessingSteps = ({ steps }) => {
  return (
    <div className="mb-4 mx-auto">
      <div className="flex items-center space-x-2 mb-3">
        <div className="w-4 h-4 bg-black flex items-center justify-center">
          <Sparkles className="w-2 h-2 text-white" />
        </div>
        <h3 className="text-xs font-medium text-black">Processing</h3>
      </div>
      <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
        {steps.map((step, index) => (
          <div key={index} className="flex items-center space-x-2 p-2 bg-gray-100">
            <div className="flex-shrink-0">
              {step.status === 'waiting' && (
                <Clock className="w-3 h-3 text-gray-400" />
              )}
              {step.status === 'processing' && (
                <Loader2 className="w-3 h-3 text-gray-600 animate-spin" />
              )}
              {step.status === 'completed' && (
                <CheckCircle className="w-3 h-3 text-black" />
              )}
              {step.status === 'error' && (
                <XCircle className="w-3 h-3 text-red-600" />
              )}
            </div>
            <span className={`text-xs truncate ${
              step.status === 'completed' ? 'text-black' : 
              step.status === 'error' ? 'text-red-600' : 
              step.status === 'processing' ? 'text-gray-600 font-medium' :
              'text-gray-500'
            }`}>
              {step.step}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProcessingSteps;