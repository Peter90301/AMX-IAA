# imports
# import cudf as pd  #run GPU version
import pandas as pd  #run CPU version
import pickle
# import cupy as np  #RUN GPU version
import numpy as np  #run CPU version



# import qiime2 as q2
# from qiime2.plugins.feature_table.methods import rarefy
# from qiime2.plugins.gemelli.actions import rpca
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import optim
import torch
import torch.nn as nn
from biom import load_table
import matplotlib.pyplot as plt
import pandas as pandas

from TRPCA import utils_intel as utils
from TRPCA import trpca_intel as trpca


def pca_pre_processing(feature_frame,pca_dimensions):
    #pre-processing &pca
    # df1=feature_frame.to_pandas()   #GPU version
    df1=feature_frame.copy()  #CPU version
    df1 = utils.clr_transformation(df1)#some issure with cupy with this function,fix later
    # df1= pd.from_pandas(df1)  #GPU version
    # feature_x=df1.to_cupy()  #GPU version
    feature_x=df1.to_numpy()  #CPU version

    print('start PCA fit')
    X1_reduced, pca1 = utils.apply_pca_partial(feature_x, pca_dimensions,train_sz=0.1)    #節省PCA時間調整train size

    return df1,pca1
     


def age_conversion(row):
    age=row['host_age']
    age_units=row['host_age_units']
    if age_units=='months'or age_units=='Months':
        final_age=age/12
    else:
        final_age=age
    return final_age



metaG_table = load_table('/home/vaquita/multiomic/RAW_DATA/redbiom/redbiom_adrc/redbiom_adrc_wolr2_fecal_v2.biom')
metaG_frame=metaG_table.to_dataframe(dense=True).transpose()
# metaG_frame=pd.from_pandas(metaG_frame)   #GPU version


metaframe=pd.read_csv('/home/vaquita/multiomic/RAW_DATA/redbiom/redbiom_adrc/redbiom_adrc_wolr2_fecal_v2.tsv',sep = '\t',dtype={'#SampleID':'object','host_age':'float','qiita_study_id':'int'}).set_index('#SampleID')



# print (metaG_frame.index[0:5])
# print (metaG_frame.columns[0:5])

# print('metadata frame info')
# print (metaframe.index[0:5])
# print (metaframe.columns[0:5])

print ( 'original metadata shape',metaframe.shape)

metaframe=metaframe.dropna(subset='host_age')#filter out nan rows

print ( 'dropped NA metadata shape',metaframe.shape)

converted_age=metaframe.apply(age_conversion,axis=1)
metaframe.insert(1,'converted_age',converted_age)

if len(metaframe)!=len(metaframe.index.duplicated(keep='first')):
    print('duplicate samples in metadata, fix datasets')

if len(metaG_frame)!=len(metaG_frame.index.duplicated(keep='first')):
    print('duplicate samples in feature table, fix datasets')

metaframe=metaframe.reindex(metaG_frame.index,axis='index')
metaframe=metaframe.dropna(subset='converted_age')#filter out nan rows
print ( 'reindexed metadata shape',metaframe.shape)

if len(metaframe)!=len(metaG_frame):
    print('warning:there are mismatch sample id in metadata and feature')



#metaG_frame=metaG_frame.reindex(AD_frame.columns,axis='columns')
pca_dimensions=512
feature_frame,pca1=pca_pre_processing(metaG_frame,pca_dimensions)




red_biom_meta=metaframe[metaframe['qiita_study_id']!=15448]   #15448 is adrc
adrc_meta=metaframe[metaframe['qiita_study_id']==15448] 


red_biom_age=red_biom_meta['converted_age']
adrc_age=adrc_meta['converted_age']

red_biom=feature_frame.reindex(red_biom_age.index,axis='index')
adrc=feature_frame.reindex(adrc_age.index,axis='index')

# feature_x=red_biom.to_cupy()    #GPU version
feature_x=red_biom.to_numpy()    #CPU version
#model_score1 = pca1.score(x_train, y_train)

feature_x=pca1.transform(feature_x)
# label_y=red_biom_age.to_cupy()   #GPU version
label_y=red_biom_age.to_numpy()   #CPU version

#generate age bins
num_bins=6
# age_bins = pandas.qcut(red_biom_age.to_pandas(), q=num_bins, labels=False, duplicates='drop') #GPU and CPU version
age_bins = pandas.qcut(red_biom_age, q=num_bins, labels=False, duplicates='drop') #GPU and CPU version

scaler=trpca.trpca_regress(feature_x, label_y,age_bins, test_size=0.2, n_dimensions=pca_dimensions, num_transformer_layers=6, epochs=500, learning_rate=1e-04, batch_size=256)


pca1_file='pca1.sav'
scaler_file='scalar.sav'
pickle.dump(scaler, open(scaler_file, 'wb'))
pickle.dump(pca1, open(pca1_file, 'wb'))




























