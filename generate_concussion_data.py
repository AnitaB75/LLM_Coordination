import json
import uuid
from datetime import datetime, timedelta

# Helper function to generate timestamps
def get_timestamp(date_str, time_of_day):
    """Convert date and time of day to Unix timestamp in milliseconds"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    if time_of_day == "AM":
        # 8 AM
        date_obj = date_obj.replace(hour=8, minute=0, second=0)
    else:
        # 6 PM
        date_obj = date_obj.replace(hour=18, minute=0, second=0)
    return int(date_obj.timestamp() * 1000)

# Function to create symptom entry
def create_symptom_entry(date_str, time_of_day, symptoms, notes=None):
    return {
        "id": str(uuid.uuid4()),
        "date": date_str,
        "timeOfDay": time_of_day,
        "timestamp": get_timestamp(date_str, time_of_day),
        "symptoms": symptoms,
        "notes": notes
    }

# Function to create screen time entry
def create_screen_time_entry(date_str, minutes):
    return {
        "date": date_str,
        "screenTimeMinutes": minutes,
        "timestamp": get_timestamp(date_str, "PM")
    }

# Generate symptoms for the period
symptoms_entries = []
screen_time_entries = []

# April 7-20: Gradual improvement
# Starting from baseline: AM (6,7,5,7,5,4,4), PM (7,8,6,7,6,5,5)
# Improvement means symptoms gradually decrease

start_date = datetime(2026, 4, 7)
end_date = datetime(2026, 5, 7)

# Baseline symptoms (from April 6)
baseline_am = {"photosensitivity": 6, "convergence": 7, "cervicogenic": 5, "headache": 7, "fatigue": 5, "back_pain": 4, "stress": 4}
baseline_pm = {"photosensitivity": 7, "convergence": 8, "cervicogenic": 6, "headache": 7, "fatigue": 6, "back_pain": 5, "stress": 5}

current_date = start_date
day_count = 0

while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    day_of_week = current_date.strftime("%A")
    is_thursday = (day_of_week == "Thursday")

    notes_am = None
    notes_pm = None

    # Determine which phase we're in
    if current_date < datetime(2026, 4, 20):
        # April 7-19: Improvement phase
        # Gradual reduction in symptoms
        improvement_factor = day_count / 13.0  # 13 days of improvement

        am_symptoms = {
            "photosensitivity": max(4, int(baseline_am["photosensitivity"] - improvement_factor * 2)),
            "convergence": max(5, int(baseline_am["convergence"] - improvement_factor * 2)),
            "cervicogenic": max(3, int(baseline_am["cervicogenic"] - improvement_factor * 2)),
            "headache": max(4, int(baseline_am["headache"] - improvement_factor * 3)),
            "fatigue": max(3, int(baseline_am["fatigue"] - improvement_factor * 2)),
            "back_pain": max(2, int(baseline_am["back_pain"] - improvement_factor * 2)),
            "stress": max(2, int(baseline_am["stress"] - improvement_factor * 2))
        }

        pm_symptoms = {
            "photosensitivity": max(5, int(baseline_pm["photosensitivity"] - improvement_factor * 2)),
            "convergence": max(6, int(baseline_pm["convergence"] - improvement_factor * 2)),
            "cervicogenic": max(4, int(baseline_pm["cervicogenic"] - improvement_factor * 2)),
            "headache": max(5, int(baseline_pm["headache"] - improvement_factor * 2)),
            "fatigue": max(4, int(baseline_pm["fatigue"] - improvement_factor * 2)),
            "back_pain": max(3, int(baseline_pm["back_pain"] - improvement_factor * 2)),
            "stress": max(3, int(baseline_pm["stress"] - improvement_factor * 2))
        }

        screen_time = 420  # Stable screen time

    elif datetime(2026, 4, 20) <= current_date < datetime(2026, 4, 24):
        # April 20-23: Bad sleep → increased headaches/eye strain
        am_symptoms = {
            "photosensitivity": 6,
            "convergence": 7,
            "cervicogenic": 5,
            "headache": 8,
            "fatigue": 6,
            "back_pain": 4,
            "stress": 5
        }

        pm_symptoms = {
            "photosensitivity": 7,
            "convergence": 8,
            "cervicogenic": 6,
            "headache": 9,
            "fatigue": 7,
            "back_pain": 5,
            "stress": 6
        }

        if current_date == datetime(2026, 4, 20):
            notes_pm = "Poor sleep last night"

        screen_time = 300  # Reduced screen time due to symptoms

    elif datetime(2026, 4, 24) <= current_date < datetime(2026, 5, 1):
        # April 24-30: Recovery from bad sleep episode
        days_since_bad_sleep = (current_date - datetime(2026, 4, 20)).days
        recovery_factor = min(1.0, days_since_bad_sleep / 7.0)

        am_symptoms = {
            "photosensitivity": max(4, int(6 - recovery_factor * 2)),
            "convergence": max(5, int(7 - recovery_factor * 2)),
            "cervicogenic": max(3, int(5 - recovery_factor * 2)),
            "headache": max(4, int(8 - recovery_factor * 4)),
            "fatigue": max(3, int(6 - recovery_factor * 3)),
            "back_pain": max(2, int(4 - recovery_factor * 2)),
            "stress": max(2, int(5 - recovery_factor * 3))
        }

        pm_symptoms = {
            "photosensitivity": max(5, int(7 - recovery_factor * 2)),
            "convergence": max(6, int(8 - recovery_factor * 2)),
            "cervicogenic": max(4, int(6 - recovery_factor * 2)),
            "headache": max(5, int(9 - recovery_factor * 4)),
            "fatigue": max(4, int(7 - recovery_factor * 3)),
            "back_pain": max(3, int(5 - recovery_factor * 2)),
            "stress": max(3, int(6 - recovery_factor * 3))
        }

        screen_time = min(420, int(300 + recovery_factor * 120))

    elif current_date == datetime(2026, 5, 1):
        # May 1: Big talk + significant sleep loss
        am_symptoms = {
            "photosensitivity": 4,
            "convergence": 5,
            "cervicogenic": 3,
            "headache": 4,
            "fatigue": 4,
            "back_pain": 2,
            "stress": 4
        }

        pm_symptoms = {
            "photosensitivity": 5,
            "convergence": 6,
            "cervicogenic": 4,
            "headache": 5,
            "fatigue": 5,
            "back_pain": 3,
            "stress": 6
        }

        notes_pm = "Big talk today - significant stress"
        screen_time = 360

    elif datetime(2026, 5, 2) <= current_date <= datetime(2026, 5, 3):
        # May 2-3: Increased symptoms and stress response, then decrease
        if current_date == datetime(2026, 5, 2):
            am_symptoms = {
                "photosensitivity": 6,
                "convergence": 7,
                "cervicogenic": 5,
                "headache": 7,
                "fatigue": 7,
                "back_pain": 4,
                "stress": 7
            }

            pm_symptoms = {
                "photosensitivity": 7,
                "convergence": 8,
                "cervicogenic": 6,
                "headache": 8,
                "fatigue": 8,
                "back_pain": 5,
                "stress": 8
            }

            notes_am = "Poor sleep from stress last night"
            screen_time = 240
        else:  # May 3
            am_symptoms = {
                "photosensitivity": 5,
                "convergence": 6,
                "cervicogenic": 4,
                "headache": 6,
                "fatigue": 6,
                "back_pain": 3,
                "stress": 6
            }

            pm_symptoms = {
                "photosensitivity": 6,
                "convergence": 7,
                "cervicogenic": 5,
                "headache": 6,
                "fatigue": 6,
                "back_pain": 4,
                "stress": 5
            }

            notes_pm = "Symptoms beginning to decrease"
            screen_time = 300

    elif datetime(2026, 5, 4) <= current_date < datetime(2026, 5, 7):
        # May 4-6: Continued improvement
        am_symptoms = {
            "photosensitivity": 4,
            "convergence": 5,
            "cervicogenic": 3,
            "headache": 5,
            "fatigue": 4,
            "back_pain": 3,
            "stress": 4
        }

        pm_symptoms = {
            "photosensitivity": 5,
            "convergence": 6,
            "cervicogenic": 4,
            "headache": 5,
            "fatigue": 5,
            "back_pain": 3,
            "stress": 4
        }

        if current_date == datetime(2026, 5, 5):
            notes_am = "NUCCA chiropractor visit"

        screen_time = 360

    else:  # May 7
        am_symptoms = {
            "photosensitivity": 3,
            "convergence": 4,
            "cervicogenic": 2,
            "headache": 3,
            "fatigue": 3,
            "back_pain": 2,
            "stress": 3
        }

        pm_symptoms = {
            "photosensitivity": 4,
            "convergence": 5,
            "cervicogenic": 3,
            "headache": 4,
            "fatigue": 4,
            "back_pain": 3,
            "stress": 3
        }

        notes_am = "Stopped wearing magnifying glasses"
        notes_pm = "Noticeable improvement after removing magnifying glasses"
        screen_time = 420

    # Add Thursday vision therapy notes
    if is_thursday:
        if notes_pm:
            notes_pm += " | Vision therapy session"
        else:
            notes_pm = "Vision therapy session"

    # Create entries
    symptoms_entries.append(create_symptom_entry(date_str, "AM", am_symptoms, notes_am))
    symptoms_entries.append(create_symptom_entry(date_str, "PM", pm_symptoms, notes_pm))
    screen_time_entries.append(create_screen_time_entry(date_str, screen_time))

    current_date += timedelta(days=1)
    if current_date < datetime(2026, 4, 20):
        day_count += 1

# Output the generated data
print(f"Generated {len(symptoms_entries)} symptom entries")
print(f"Generated {len(screen_time_entries)} screen time entries")

# Save to JSON files for review
with open('/Users/calypsocalcott/LLM_Coordination/new_symptoms.json', 'w') as f:
    json.dump(symptoms_entries, f, indent=2)

with open('/Users/calypsocalcott/LLM_Coordination/new_screen_time.json', 'w') as f:
    json.dump(screen_time_entries, f, indent=2)

print("\nSaved new entries to new_symptoms.json and new_screen_time.json")
