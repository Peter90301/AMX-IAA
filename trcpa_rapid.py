# imports
import cudf as pd
import pickle
import cupy as np
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

from TRPCA import utils
from TRPCA import trpca


def pre_processing(feature_frame,meta_frame,pca_dimensions):
    #pre-processing &pca
    df1=feature_frame.to_pandas()
    df1 = utils.clr_transformation(df1)#some issure with cupy with this function,fix later
    df1= pd.from_pandas(df1)
    feature_x=df1.to_cupy()

    label_y=meta_frame.to_cupy()
    print('start PCA fit')
    X1_reduced, pca1 = utils.apply_pca_partial(feature_x, pca_dimensions,0.4)
    return X1_reduced,label_y,pca1
     


def age_conversion(row):
    age=row['host_age']
    age_units=row['host_age_units']
    if age_units=='months'or age_units=='Months':
        final_age=age/12
    else:
        final_age=age
    return final_age



metaG_table = load_table('/home/zheyu/multiomic/RAW_DATA/redbiom_wolr2_fecal.biom')
metaG_frame=metaG_table.to_dataframe(dense=True).transpose()
metaG_frame=pd.from_pandas(metaG_frame)

AD_table = load_table('/home/zheyu/multiomic/RAW_DATA/ADRC_FULL/adrc_coverage_filtered_table.biom')
AD_frame=AD_table.to_dataframe(dense=True).transpose()
AD_frame=pd.from_pandas(AD_frame)



metaframe=pd.read_csv('/home/zheyu/multiomic/RAW_DATA/redbiom_wolr2_fecal.minimal.tsv',sep = '\t',dtype={'#SampleID':'object','host_age':'float'}).set_index('#SampleID')



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
metaframe=metaframe['converted_age']

#apply CLR transform&PCA
num_bins=6

#metaG_frame=metaG_frame.reindex(AD_frame.columns,axis='columns')

feature_x,label_y,pca1=pre_processing(metaG_frame,metaframe,512)

age_bins = pandas.qcut(metaframe.to_pandas(), q=num_bins, labels=False, duplicates='drop')


print('final shape',label_y.shape)
print('final shape',feature_x.shape)

scaler=trpca.trpca_regress(feature_x, label_y,age_bins, test_size=0.2, n_dimensions=512, num_transformer_layers=6, epochs=500, learning_rate=1e-04, batch_size=256)

pca1_file='pca1.sav'
scaler_file='scalar.sav'
pickle.dump(scaler, open(scaler_file, 'wb'))
pickle.dump(pca1, open(pca1_file, 'wb'))




























