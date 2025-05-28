import React from 'react';

const EntitiesModal = ({ isOpen, onClose, entities }) => {
  if (!isOpen) return null;

  // Filter entities to include only those with present values
  const filteredEntities = Object.entries(entities).reduce((acc, [type, value]) => {
    if (Array.isArray(value) && value.length > 0) {
      value.forEach((val, index) => {
        if (val) acc.push({ type: `${type} ${index + 1}`, value: val });
      });
    } else if (value !== null && value !== undefined && value !== '') {
      acc.push({ type, value });
    }
    return acc;
  }, []);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full mx-4 overflow-hidden">
        <div className="px-8 py-6 border-b border-gray-100">
          <h2 className="text-xl font-light text-gray-900 tracking-wide">Extracted Entities</h2>
        </div>
        
        <div className="px-8 py-6 max-h-96 overflow-y-auto">
          {filteredEntities.length > 0 ? (
            <div className="space-y-4">
              {filteredEntities.map((entity, index) => (
                <div key={index} className="group">
                  <div className="flex flex-col space-y-1">
                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                      {entity.type}
                    </span>
                    <span className="text-gray-900 font-light leading-relaxed">
                      {entity.value}
                    </span>
                  </div>
                  {index < filteredEntities.length - 1 && (
                    <div className="mt-4 h-px bg-gradient-to-r from-transparent via-gray-200 to-transparent"></div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
                <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-gray-500 font-light">No entities found</p>
            </div>
          )}
        </div>
        
        <div className="px-8 py-6 bg-gray-50 border-t border-gray-100">
          <button
            onClick={onClose}
            className="w-full px-6 py-3 bg-gray-900 text-white rounded-xl font-light tracking-wide hover:bg-gray-800 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-gray-900 focus:ring-offset-2"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default EntitiesModal;