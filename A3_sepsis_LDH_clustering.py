# clustering
import pickle
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering

data_folder = '/Users/xuzhenxing/Documents/Sepsis/sepsis_manuscript/LDH/result/'

result_fold = data_folder

num_cluster = 4 # number of clusters

linkage = 'complete'

with open(data_folder +'SOFA_sample.pkl', 'rb') as f:
    Id_dis = pickle.load(f)

ICU_stay_ID = Id_dis[0]
dis_matrix = Id_dis[1]

all_ICU_IDs = ICU_stay_ID

model = AgglomerativeClustering(n_clusters=num_cluster, affinity='precomputed', linkage=linkage)
model.fit(dis_matrix)
labels = model.labels_
labels_list = list(labels)

cluster_id_0 = []
cluster_id_1 = []
cluster_id_2 = []
cluster_id_3 = []

for i in range(len(labels_list)):
    if labels_list[i] == 0:
        cluster_id_0.append(all_ICU_IDs[i])
    if labels_list[i] == 1:
        cluster_id_1.append(all_ICU_IDs[i])
        # print all_ICU_IDs[i]
    if labels_list[i] == 2:
        cluster_id_2.append(all_ICU_IDs[i])
        # print all_ICU_IDs[i]
    if labels_list[i] == 3:
        cluster_id_3.append(all_ICU_IDs[i])

print len(cluster_id_0),len(cluster_id_1),len(cluster_id_2),len(cluster_id_3)

# save clustering results

with open(result_fold+'cluster_id_0.pkl', 'wb') as f_0:
    pickle.dump(cluster_id_0, f_0)
with open(result_fold+'cluster_id_1.pkl', 'wb') as f_1:
    pickle.dump(cluster_id_1, f_1)
with open(result_fold+'cluster_id_2.pkl', 'wb') as f_2:
    pickle.dump(cluster_id_2, f_2)
with open(result_fold+'cluster_id_3.pkl', 'wb') as f_3:
    pickle.dump(cluster_id_3, f_3)


    
    
    
    
