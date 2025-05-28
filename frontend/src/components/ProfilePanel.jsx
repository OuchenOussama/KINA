import React from 'react';
import { Shield } from 'lucide-react';

const ProfilePanel = ({ userProfile, setUserProfile }) => {
  return (
    <div className="mb-4 p-4 bg-gray-100">
      <div className="flex items-center space-x-2 mb-4">
        <Shield className="w-4 h-4 text-black" />
        <h3 className="text-sm font-medium text-black">Safety Profile</h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-xs text-black mb-1">Age</label>
          <input
            type="number"
            value={userProfile.age || ''}
            onChange={(e) => setUserProfile((prev) => ({ ...prev, age: parseInt(e.target.value) || null }))}
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
            placeholder="Age"
          />
        </div>
        <div>
          <label className="block text-xs text-black mb-1">Pregnancy</label>
          <select
            value={userProfile.isPregnant}
            onChange={(e) => setUserProfile((prev) => ({ ...prev, isPregnant: e.target.value === 'true' }))}
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
          >
            <option value={false}>No</option>
            <option value={true}>Yes</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-black mb-1">Allergies</label>
          <input
            type="text"
            value={userProfile.allergies.join(', ')}
            onChange={(e) =>
              setUserProfile((prev) => ({
                ...prev,
                allergies: e.target.value.split(',').map((a) => a.trim()).filter((a) => a),
              }))
            }
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
            placeholder="Allergies"
          />
        </div>
      </div>
    </div>
  );
};

export default ProfilePanel;