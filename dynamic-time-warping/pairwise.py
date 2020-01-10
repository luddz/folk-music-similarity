import sys
import os
import json
import importlib
if sys.version_info < (3,0):
    import cPickle as pickle
else:
    import pickle
import numpy as np
import argparse
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--json_file_1')
parser.add_argument('--json_file_2')

args = parser.parse_args()

metadata_path = "config5-wrepeats-20160112-222521.pkl"
temperature = args.temperature
json_file_1 = args.json_file_1
json_file_2 = args.json_file_2


with open(json_file_1) as annotation_file_1:
    annotations_1 = json.load(annotation_file_1)

with open(json_file_2) as annotation_file_2:
    annotations_2 = json.load(annotation_file_2)

if not os.path.isdir('matrices'):
    os.makedirs('matrices')

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

token2idx = metadata['token2idx']

start_idx = token2idx['<s>']

def sigmoid(x): return 1/(1 + np.exp(-x))

def softmax(x,T): 
    expx=np.exp(x/T)
    sumexpx=np.sum(expx)
    if sumexpx==0:
        maxpos=x.argmax()
        x=np.zeros(x.shape, dtype=x.dtype)
        x[0][maxpos]=1
    else:
        x=expx/sumexpx
    return x

def create_matrix(annotation_object):
    annotation = "M:" + annotation_object['meter'] + " " + "K:" + annotation_object['key'] + " " + annotation_object['transcription']

    LSTM_Wxi=[]
    LSTM_Wxf=[]
    LSTM_Wxc=[]
    LSTM_Wxo=[]
    LSTM_Whi=[]
    LSTM_Whf=[]
    LSTM_Whc=[]
    LSTM_Who=[]
    LSTM_bi=[]
    LSTM_bf=[]
    LSTM_bc=[]
    LSTM_bo=[]
    LSTM_cell_init=[]
    LSTM_hid_init=[]
    htm1=[]
    ctm1=[]

    numlayers=3 # hard coded for now, but this should be saved in the model pickle
    for jj in range(numlayers):
        LSTM_Wxi.append(metadata['param_values'][2+jj*14-1])
        LSTM_Whi.append(metadata['param_values'][3+jj*14-1])
        LSTM_bi.append(metadata['param_values'][4+jj*14-1])
        LSTM_Wxf.append(metadata['param_values'][5+jj*14-1])
        LSTM_Whf.append(metadata['param_values'][6+jj*14-1])
        LSTM_bf.append(metadata['param_values'][7+jj*14-1])
        LSTM_Wxc.append(metadata['param_values'][8+jj*14-1])
        LSTM_Whc.append(metadata['param_values'][9+jj*14-1])
        LSTM_bc.append(metadata['param_values'][10+jj*14-1])
        LSTM_Wxo.append(metadata['param_values'][11+jj*14-1])
        LSTM_Who.append(metadata['param_values'][12+jj*14-1])
        LSTM_bo.append(metadata['param_values'][13+jj*14-1])
        LSTM_cell_init.append(metadata['param_values'][14+jj*14-1])
        LSTM_hid_init.append(metadata['param_values'][15+jj*14-1])
        htm1.append(LSTM_hid_init[jj])
        ctm1.append(LSTM_cell_init[jj])

    FC_output_W = metadata['param_values'][43];
    FC_output_b = metadata['param_values'][44];

    sizeofx=LSTM_Wxi[0].shape[0]

    x = np.zeros(sizeofx, dtype=np.int8)
    # Converting the seed passed as an argument into a list of idx
    annotation_sequence = [start_idx]
    distribution_matrix=[]
    for token in annotation.split(' '):
        annotation_sequence.append(token2idx[token])
            
    # Running the annotation through the network
    for tok in annotation_sequence[:-1]:
        x = np.zeros(sizeofx, dtype=np.int8)
        x[tok] = 1
        for jj in range(numlayers):
            it=sigmoid(np.dot(x,LSTM_Wxi[jj]) + np.dot(htm1[jj],LSTM_Whi[jj]) + LSTM_bi[jj])
            ft=sigmoid(np.dot(x,LSTM_Wxf[jj]) + np.dot(htm1[jj],LSTM_Whf[jj]) + LSTM_bf[jj])
            ct=np.multiply(ft,ctm1[jj]) + np.multiply(it,np.tanh(np.dot(x,LSTM_Wxc[jj]) + np.dot(htm1[jj],LSTM_Whc[jj]) + LSTM_bc[jj]))
            ot=sigmoid(np.dot(x,LSTM_Wxo[jj]) + np.dot(htm1[jj],LSTM_Who[jj]) + LSTM_bo[jj])
            ht=np.multiply(ot,np.tanh(ct))
            x=ht
            ctm1[jj]=ct
            htm1[jj]=ht
        
        #Prop distribution vector for next token 
        distribution_for_next_token = softmax(np.dot(x,FC_output_W) + FC_output_b,temperature)
        distribution_matrix.append(distribution_for_next_token[-1].squeeze())


    #To numpy matrix
    return np.array(distribution_matrix)

matrices_first = []
matrices_second = []

for annotation_object in annotations_1:
    annotation_id = annotation_object['id']

    matrix = create_matrix(annotation_object)
    matrices_first.append((annotation_id, matrix))

for annotation_object in annotations_2:
    annotation_id = annotation_object['id']
    
    matrix = create_matrix(annotation_object)
    matrices_second.append((annotation_id, matrix))


def compare_two_matrices(matrix_1, matrix_2):
    distance, path = fastdtw(np.array(matrix_1), np.array(matrix_2), dist=euclidean)
    np_1 = np.array(matrix_1)
    np_2 = np.array(matrix_2)
    max_length = matrix_1.shape[0] if matrix_1.shape[0] > matrix_2.shape[0] else matrix_2.shape[0]
    norm_distance = (max_length - distance) / max_length

    return abs(norm_distance)

metrics = []
for (id1, matrix_1) in matrices_first:
    print("Distances for id: %d " %(id1))
    for (id2, matrix_2) in matrices_second:
        norm_distance = compare_two_matrices(matrix_1, matrix_2)
        metrics.append({'id1': id1, 'id2': id2, 'metric': norm_distance})

metrics.sort(reverse=True, key=lambda x: x['metric'])
metrics.sort(key=lambda x: x['id1'])

with open("output2.json", 'w') as fout:
    json.dump(metrics, fout, ensure_ascii=False)