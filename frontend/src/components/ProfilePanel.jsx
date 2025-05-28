import React from 'react';
import { Shield } from 'lucide-react';

const ProfilePanel = ({ userProfile, setUserProfile }) => {
  return (
    <div className="mb-4 p-4 bg-gray-100">
      <div className="flex items-center space-x-2 mb-4">
        <Shield className="w-4 h-4 text-black" />
        <h3 className="text-sm font-medium text-black">Safety Profile</h3>
      </div>
      
      {/* Basic Information Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-xs text-black mb-1">Age</label>
          <input
            type="number"
            max={100}
            min={1}
            value={userProfile.age || ''}
            onChange={(e) => setUserProfile((prev) => ({ ...prev, age: parseInt(e.target.value) || null }))}
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
            placeholder="Age"
          />
        </div>
        <div>
          <label className="block text-xs text-black mb-1">Weight (kg)</label>
          <input
            type="number"
            max={300}
            min={1}
            value={userProfile.weight || ''}
            onChange={(e) => setUserProfile((prev) => ({ ...prev, weight: parseInt(e.target.value) || null }))}
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
            placeholder="Weight"
          />
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
            placeholder="e.g. peanuts, shellfish"
          />
        </div>
      </div>

      {/* Pregnancy and Breastfeeding Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
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
          <label className="block text-xs text-black mb-1">Breastfeeding</label>
          <select
            value={userProfile.isBreastfeeding}
            onChange={(e) => setUserProfile((prev) => ({ ...prev, isBreastfeeding: e.target.value === 'true' }))}
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
          >
            <option value={false}>No</option>
            <option value={true}>Yes</option>
          </select>
        </div>
      </div>

      {/* Medical Information Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-xs text-black mb-1">Medical Conditions</label>
          <input
            type="text"
            value={userProfile.medicalConditions.join(', ')}
            onChange={(e) =>
              setUserProfile((prev) => ({
                ...prev,
                medicalConditions: e.target.value.split(',').map((c) => c.trim()).filter((c) => c),
              }))
            }
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
            placeholder="e.g. diabetes, hypertension"
          />
        </div>
        <div>
          <label className="block text-xs text-black mb-1">Current Medications</label>
          <input
            type="text"
            value={userProfile.currentMedications.join(', ')}
            onChange={(e) =>
              setUserProfile((prev) => ({
                ...prev,
                currentMedications: e.target.value.split(',').map((m) => m.trim()).filter((m) => m),
              }))
            }
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
            placeholder="e.g. aspirin, metformin"
          />
        </div>
      </div>

      {/* Organ Function Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-xs text-black mb-1">Kidney Function</label>
          <select
            value={userProfile.kidneyFunction}
            onChange={(e) => setUserProfile((prev) => ({ ...prev, kidneyFunction: e.target.value }))}
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
          >
            <option value="normal">Normal</option>
            <option value="mild">Mild Impairment</option>
            <option value="moderate">Moderate Impairment</option>
            <option value="severe">Severe Impairment</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-black mb-1">Liver Function</label>
          <select
            value={userProfile.liverFunction}
            onChange={(e) => setUserProfile((prev) => ({ ...prev, liverFunction: e.target.value }))}
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
          >
            <option value="normal">Normal</option>
            <option value="mild">Mild Impairment</option>
            <option value="moderate">Moderate Impairment</option>
            <option value="severe">Severe Impairment</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-black mb-1">Heart Condition</label>
          <select
            value={userProfile.heartCondition}
            onChange={(e) => setUserProfile((prev) => ({ ...prev, heartCondition: e.target.value === 'true' }))}
            className="w-full px-2 py-1 border border-gray-300 text-sm bg-white text-black"
          >
            <option value={false}>No</option>
            <option value={true}>Yes</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default ProfilePanel;