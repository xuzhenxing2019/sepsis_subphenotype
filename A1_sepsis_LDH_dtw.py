import os
import pandas as pd
import numpy as np
from time import time
from tslearn.metrics import cdist_dtw
import matplotlib
matplotlib.use('TkAgg')
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import seaborn as sns; sns.set(color_codes=True)
import pickle


data_fold = '/Users/xuzhenxing/Documents/Sepsis/sepsis_manuscript/LDH/'
result_fold = '/Users/xuzhenxing/Documents/Sepsis/sepsis_manuscript/LDH/result/'

data_type = 'SOFA_sample'
data = pd.read_csv(os.path.join(data_fold,data_type+'.csv'),index_col=0)

ICUstay_ID_s = list(data.index)
score_normalized = data.sub(data.min(axis=1), axis=0).div(data.max(axis=1) - data.min(axis=1), axis=0).fillna(0)
score_normalized = np.array(score_normalized)

print('running dtw....')
start = time()
dis = cdist_dtw(score_normalized,n_jobs=4)

c_time = time() - start
print(c_time)

score_id = [ICUstay_ID_s,dis]


with open(os.path.join(result_fold,data_type+'.pkl'),'wb') as f:
    pickle.dump(score_id,f)

# obtaining clustermap figure
linkage = hc.linkage(sp.distance.squareform(dis,checks=False), method='complete')
ns_plot = sns.clustermap(dis, row_linkage=linkage, col_linkage=linkage,standard_scale=1)
ns_plot.savefig(result_fold+'/'+'clustermap_'+ data_type +".png", dpi = 600)












