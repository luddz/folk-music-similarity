import json
import sys

with open('./songs.json') as json_file:
    data = json.load(json_file)
    for p in data:
        print('X:' + str(p['id']))
        print('T:' + str(p['id']))
        print('M:' + p['meter'])
        print('K:' + p['key'])
        print(p['transcription'].replace(" ", ""))
        print('')