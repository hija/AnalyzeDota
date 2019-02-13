import dota2api
import os
import pprint
import time
import json

ACCOUNTS = {
    "Hilko": 64989974
}

api = dota2api.Initialise(os.environ['D2_API_KEY'])
hist = api.get_match_history(account_id=ACCOUNTS["Hilko"])

# Get matchids from history object
match_ids = [match['match_id'] for match in hist['matches']]

for i in range(10):
    min_match_id = min(match_ids) - 1
    print(f'Starting @{min_match_id}')

    hist = api.get_match_history(account_id=ACCOUNTS["Hilko"], start_at_match_id = min_match_id)
    match_ids.extend([match['match_id'] for match in hist['matches']])
    match_ids = list(set(match_ids))

    print('Current matches:', len(match_ids))
    time.sleep(5)

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

# Get further information about matches
match_details = {}

for match_id in match_ids:
    print(f'>> Downloading {match_id}')
    ## Download match details
    time.sleep(7)
    match_details = api.get_match_details(match_id=match_id)
    with open(f'data/{match_id}.json', 'w') as jsonf:
        json.dump(match_details, jsonf)
