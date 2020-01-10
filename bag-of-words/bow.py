from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

my_map = {"0": "d'", "1": "=A,", "2": "^c'", "3": "=e", "4": "=d", "5": "=g", "6": "=f", "7": "=a", "8": "=c", "9": "=b", "10": "_G", "11": "_E", "12": "_D", "13": "_C", "14": "_B", "15": "_A", "16": "2<", "17": "2>", "18": "=E", "19": "=D", "20": "_B,", "21": "=F", "22": "=A", "23": "4", "24": "=C", "25": "=B", "26": "_g", "27": "8", "28": "_e", "29": "_d", "30": "_c", "31": "<", "32": "_a", "33": "(9", "34": "|2", "35": "D", "36": "|1", "37": "(2", "38": "(3", "39": "|:", "40": "(7", "41": "(4", "42": "(5", "43": ":|", "44": "9", "45": "3/2", "46": "3/4", "47": "=f'", "48": "2", "49": "d", "50": "_E,", "51": "B,", "52": "16", "53": "|", "54": "^A,", "55": "b'", "56": "_e'", "57": "M:9/8", "58": "E,", "59": "</s>", "60": "3", "61": "7", "62": "^F,", "63": "=G,", "64": "C", "65": "G", "66": "e'", "67": "_d'", "68": "^f'", "69": "[", "70": "b", "71": "c", "72": "z", "73": "g", "74": "^G,", "75": "=F,", "76": "K:Cmin", "77": "K:Cmix", "78": "=c'", "79": "C,", "80": "<s>", "81": "]", "82": "=G", "83": "M:12/8", "84": "6", "85": "=E,", "86": "K:Cmaj", "87": ">", "88": "B", "89": "F", "90": "c'", "91": "^C,", "92": "5/2", "93": "G,", "94": "f", "95": "=e'", "96": "_b", "97": "_A,", "98": "F,", "99": "/2>", "100": "/2<", "101": "f'", "102": "M:6/8", "103": "4>", "104": "M:4/4", "105": "A,", "106": "M:2/4", "107": "=C,", "108": "5", "109": "M:3/4", "110": "12", "111": "M:3/2", "112": "K:Cdor", "113": "A", "114": "E", "115": "a'", "116": "(6", "117": "^A", "118": "^C", "119": "^D", "120": "^F", "121": "^G", "122": "a", "123": "g'", "124": "D,", "125": "/4", "126": "e", "127": "/3", "128": "7/2", "129": "=B,", "130": "/8", "131": "^a", "132": "^c", "133": "^d", "134": "/2", "135": "^f", "136": "^g"}

inv_map = {v: k for k, v in my_map.items()}

class BOW:
  def __init__(self, ngram_range=(2,8)):
    self.tfidf = TfidfVectorizer(token_pattern = r"\S+", lowercase=False, ngram_range=ngram_range)

  def fit(self, json):
    data = [item['transcription'] for item in json]
    self.tfidf.fit(data)
    
  def compare(self, json_x, json_y):
    x = [item['transcription'] for item in np.array(json_x, ndmin=1)]
    x_ids = [item['id'] for item in np.array(json_x, ndmin=1)]
    y = [item['transcription'] for item in np.array(json_y, ndmin=1)]
    y_ids = [item['id'] for item in np.array(json_y, ndmin=1)]
    sim = cosine_similarity(self.tfidf.transform(x), self.tfidf.transform(y))
    ret = []
    for x_idx, x_id in enumerate(x_ids):
      for y_idx, y_id in enumerate(y_ids):
        if x_id != y_id:
          ret.append((x_id,y_id,sim[x_idx][y_idx]))
    return json.dumps(ret)

def get_ranks(bow, tests, data):
  ranks = []
  for i, test in enumerate(tests):
    for j, item in enumerate(reversed(sorted(json.loads(bow.compare(test,data)), key=lambda x: x[2]))):
      #print(str(item[0])[:-1])
      if str(item[0])[:-1] == str(item[1]) or str(item[0])[1:] == str(item[1]):
        ranks.append((i+1, j+1, item[2]))
        break
  return ranks