import json

# Load the original data
with open('/Users/calypsocalcott/Documents/GitHub/concussion-tracker/public/data.json', 'r') as f:
    data = json.load(f)

# Load the new entries
with open('/Users/calypsocalcott/LLM_Coordination/new_symptoms.json', 'r') as f:
    new_symptoms = json.load(f)

with open('/Users/calypsocalcott/LLM_Coordination/new_screen_time.json', 'r') as f:
    new_screen_time = json.load(f)

# Add new entries to the data
data['symptoms'].extend(new_symptoms)
data['rescueTime'].extend(new_screen_time)

# Save updated data back to the file
with open('/Users/calypsocalcott/Documents/GitHub/concussion-tracker/public/data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Updated data.json:")
print(f"  Total symptom entries: {len(data['symptoms'])}")
print(f"  Total screen time entries: {len(data['rescueTime'])}")
print(f"  Date range: {data['symptoms'][0]['date']} to {data['symptoms'][-1]['date']}")
