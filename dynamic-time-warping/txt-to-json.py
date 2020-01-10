import json
import argparse

parser = argparse.ArgumentParser()
arser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="training-data-small.txt")
args = parser.parse_args()
data_file = args.file

songs = []
with open(data_file, 'r', encoding="utf-8") as f:
    id = 0
    song_object = dict()
    for line in f:
        line = line.strip()
        if len(line) != 0:
            if line[0] == 'T':
                song_object['title'] = line[2:]
            elif line[0] == 'M':
                song_object['meter'] = line[2:]
            elif line[0] == 'K':
                song_object['key'] = line[2:]
            else:
                song_object['transcription'] = line
                song_object['id'] = id
                songs.append(song_object)
                id += 1
                song_object = dict()
    


with open(data_file.split('.')[0] + ".json", 'w', encoding='utf8') as fout:
    json.dump(songs, fout, ensure_ascii=False)