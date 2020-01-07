import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist




#in_data = arff.loadarff('./bestTest/100in/extracted_feature_values.arff')
#out_data = arff.loadarff('./bestTest/100out/extracted_feature_values.arff')
#bsin_data = arff.loadarff('./bestTest/bsin/extracted_feature_values.arff')

ses_df = pd.read_csv("./session/feature_vals.csv") 
gen_df = pd.read_csv("./gen/feature_vals.csv") 
test_df = pd.read_csv("./test/feature_vals.csv") 

ses_df.rename(columns={'Unnamed: 0':'song_id'}, inplace=True)
gen_df.rename(columns={'Unnamed: 0':'song_id'}, inplace=True)
test_df.rename(columns={'Unnamed: 0':'song_id'}, inplace=True)
base_df = pd.concat([ses_df, gen_df], axis=0, ignore_index=True)

for row in range(test_df.shape[0]):
    test_df.at[row, 'song_id'] = re.search('[0-9]+', test_df.at[row, 'song_id']).group(0)

for row in range(base_df.shape[0]):
    base_df.at[row, 'song_id'] = re.search('[0-9]+', base_df.at[row, 'song_id']).group(0)

scaler = StandardScaler()
# Separating out the features
base_x = base_df.loc[:, base_df.columns != 'song_id'].values# Separating out the target
base_y = base_df.loc[:,['song_id']].values# Standardizing the features
base_x = scaler.fit_transform(base_x)

test_x = test_df.loc[:, test_df.columns != 'song_id'].values# Separating out the target
test_y = test_df.loc[:,['song_id']].values# Standardizing the features
test_x = scaler.transform(test_x)


pca = PCA(n_components=0.95, svd_solver = 'full')
base_pc = pca.fit_transform(base_x)
test_pc = pca.transform(test_x)


#Pretty packaging
dists = cdist(test_pc, base_pc, 'cosine')

test_ids = ['241231', '359381', '147869', '110078', '273491', '136891', '140522', '145595', '111000', '129506']
base_ids = ['24123', '35938', '47869', '10078', '27349', '36891', '40522', '45595', '11000', '29506']

dist_dic = {};
rank_dic = {};
for i in range(dists.shape[0]):
    dist_dic[test_y.flatten()[i]] = dict(zip(base_y.flatten(), dists[i]))
    rank = np.argsort(np.argsort(dists[i])) + 1
    rank_dic[test_y.flatten()[i]] = dict(zip(base_y.flatten(), rank))

for i in range(len(test_ids)):
    print('Test id:', test_ids[i])
    print('Data id:', base_ids[i])
    print('Rank:', rank_dic[test_ids[i]][base_ids[i]])
    print('Dist:', dist_dic[test_ids[i]][base_ids[i]])
    print('')
#for t in test_ids: