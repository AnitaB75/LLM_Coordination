import json
from datetime import datetime

def calculate_nerve_static(date_str, time_of_day):
    """
    Calculate nerve_static value based on date.
    Pattern:
    - Jan 5: 10/10 (peak)
    - Slowly decreases until March 23
    - March 23: Spike to ~9
    - Decreases
    - April 20: Spike to ~7 (decreases more quickly)
    - May 1: Spike to ~6 (decreases even more quickly)
    - May 7: Lower (~3-4)
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")

    jan5 = datetime(2026, 1, 5)
    feb6 = datetime(2026, 2, 6)
    mar23 = datetime(2026, 3, 23)
    apr20 = datetime(2026, 4, 20)
    may1 = datetime(2026, 5, 1)
    may7 = datetime(2026, 5, 7)

    # Calculate days from reference points
    days_from_jan5 = (date - jan5).days

    if date < feb6:
        # Before our data range (but won't happen with current data)
        # Jan 5 = 10, decreasing to Feb 6
        days = (date - jan5).days
        return max(6, 10 - int(days / 8))  # Slow decrease

    elif feb6 <= date < mar23:
        # Feb 6 to Mar 23: Slow decrease from ~6-7 to ~4-5
        days = (date - feb6).days
        total_days = (mar23 - feb6).days
        progress = days / total_days
        base = 7 - (progress * 2)  # Decrease from 7 to 5

        # AM/PM variation
        if time_of_day == "AM":
            return max(4, int(base - 0.5))
        else:
            return max(5, int(base))

    elif date == mar23:
        # March 23: Spike
        if time_of_day == "AM":
            return 8
        else:
            return 9

    elif mar23 < date < datetime(2026, 4, 3):
        # Mar 24 - Apr 2: Quick decrease from spike
        days = (date - mar23).days
        base = 9 - (days * 0.6)  # Decreases 0.6 per day

        if time_of_day == "AM":
            return max(4, int(base - 1))
        else:
            return max(5, int(base))

    elif datetime(2026, 4, 3) <= date < apr20:
        # Apr 3 - Apr 19: Stable low
        if time_of_day == "AM":
            return 4
        else:
            return 5

    elif date == apr20:
        # April 20: Spike (smaller than Mar 23)
        if time_of_day == "AM":
            return 6
        else:
            return 7

    elif apr20 < date < may1:
        # Apr 21 - Apr 30: Faster decrease than after Mar 23 spike
        days = (date - apr20).days
        base = 7 - (days * 0.8)  # Decreases 0.8 per day (faster)

        if time_of_day == "AM":
            return max(3, int(base - 1))
        else:
            return max(4, int(base))

    elif date == may1:
        # May 1: Spike (even smaller)
        if time_of_day == "AM":
            return 5
        else:
            return 6

    elif may1 < date <= may7:
        # May 2 - May 7: Very fast decrease
        days = (date - may1).days
        base = 6 - (days * 1.0)  # Decreases 1.0 per day (very fast)

        if time_of_day == "AM":
            return max(2, int(base - 1))
        else:
            return max(2, int(base))

    else:
        # After May 7
        if time_of_day == "AM":
            return 2
        else:
            return 2


# Load the data
with open('/Users/calypsocalcott/Documents/GitHub/concussion-tracker/public/data.json', 'r') as f:
    data = json.load(f)

# Add nerve_static to each symptom entry
for entry in data['symptoms']:
    nerve_static_value = calculate_nerve_static(entry['date'], entry['timeOfDay'])
    entry['symptoms']['nerve_static'] = nerve_static_value

# Save back
with open('/Users/calypsocalcott/Documents/GitHub/concussion-tracker/public/data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Added nerve_static to {len(data['symptoms'])} symptom entries")

# Show key dates
print("\nNerve static values at key dates:")
key_dates = [
    '2026-02-06', '2026-03-01', '2026-03-23', '2026-03-24',
    '2026-04-10', '2026-04-20', '2026-04-21', '2026-05-01',
    '2026-05-02', '2026-05-07'
]

for date in key_dates:
    entries = [s for s in data['symptoms'] if s['date'] == date]
    if entries:
        am_val = next((e['symptoms']['nerve_static'] for e in entries if e['timeOfDay'] == 'AM'), None)
        pm_val = next((e['symptoms']['nerve_static'] for e in entries if e['timeOfDay'] == 'PM'), None)
        print(f"  {date}: AM={am_val}, PM={pm_val}")
