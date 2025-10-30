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
from cuml.decomposition import IncrementalPCA
import xgboost
import sklearn 
import matplotlib.pyplot as plt

from TRPCA import utils
from TRPCA import transfer
def age_conversion(row):
    age=row['host_age']
    age_units=row['host_age_units']
    if age_units=='months'or age_units=='Months':
        final_age=age/12
    else:
        final_age=age
    return final_age

def diagnosis_conversion(row):
    diagnosis=row['diagnosis']
    if diagnosis=='Normal cognition':
        return 0
    else:
        return 1


def pre_processing(feature_frame,meta_frame,pca_pretrained_path,pre_trained_scalar_path):
    #pre-processing &pca
    df1=feature_frame.to_pandas()
    df1 = utils.clr_transformation(df1)#some issure with cupy with this function,fix later
    df1= pd.from_pandas(df1)
    feature_x=df1.to_cupy()
    label_y=meta_frame.to_cupy()
    print('start PCA fit')
    with open(pca_pretrained_path, 'rb') as f:
        pca = pickle.load(f)
    with open(pre_trained_scalar_path, 'rb') as f2:
        scalar = pickle.load(f2)
    X1_reduced=pca.transform(feature_x)
    X1_reduced=scalar.transform(X1_reduced)
    return X1_reduced,label_y
     


metaG_table = load_table('/home/zheyu/multiomic/RAW_DATA/redbiom/redbiom_adrc/redbiom_adrc_wolr2_fecal_v2.biom')
metaG_frame=metaG_table.to_dataframe(dense=True).transpose()
metaG_frame=pd.from_pandas(metaG_frame)

metaframe=pd.read_csv('/home/zheyu/multiomic/RAW_DATA/redbiom/redbiom_adrc/redbiom_adrc_wolr2_fecal_v2.tsv',sep = '\t',dtype={'#SampleID':'object','host_age':'float','qiita_study_id':'int','diagnosis':'object','apoe':'object'}).set_index('#SampleID')
adrc_meta=metaframe[metaframe['qiita_study_id']==15448]
converted_age=adrc_meta.apply(age_conversion,axis=1)

adrc_meta.insert(1,'converted_age',converted_age)
AD_metaframe=adrc_meta
adrc=metaG_frame.reindex(adrc_meta.index,axis='index')
AD_frame=adrc


print(AD_frame.shape)
print(AD_metaframe.shape)


#AD_metaframe =AD_metaframe[AD_metaframe['converted_age']>60]

""" AD_diagno_meta=AD_metaframe[AD_metaframe['diagnosis'].isin(['Dementia'])]
AD_normal_meta=AD_metaframe[AD_metaframe['diagnosis']=='Normal cognition']


AD_normal_age=AD_normal_meta['converted_age']
AD_diagno_age=AD_diagno_meta['converted_age']


AD_normal=AD_frame.reindex(AD_normal_age.index,axis='index')
AD_diagno=AD_frame.reindex(AD_diagno_age.index,axis='index')

print(AD_diagno_meta.shape)
print(AD_normal_meta.shape) """

pre_trained_model_path='/home/zheyu/temp_model_save/best_designated_skin_age_regression_model.pth'
pca_path='pca1.sav'
scalar_path='scalar.sav'
pca_dimensions=512
fine_tuned_path='/home/zheyu/temp_model_save/best_designated_skin_age_regression_model_finetuned.pth'

""" feature_x,label_y=pre_processing(AD_normal,AD_normal_age,pca_path,scalar_path)
transfer.fine_tuning(feature_x, label_y, test_size=0.2, n_dimensions=pca_dimensions, num_transformer_layers=6, epochs=300, learning_rate=1e-05, batch_size=32,pre_trained_model_path=pre_trained_model_path, fine_tuned_model_path=fine_tuned_path)


feature_x,label_y=pre_processing(AD_normal,AD_normal_age,pca_path,scalar_path)
test_actuals_normal,test_preds_normal=transfer.age_regressor_validate(feature_x,label_y,fine_tuned_path,n_dimensions=pca_dimensions, num_transformer_layers=6)

feature_x,label_y=pre_processing(AD_diagno,AD_diagno_age,pca_path,scalar_path)
test_actuals_abnormal,test_preds_abnormal=transfer.age_regressor_validate(feature_x,label_y,fine_tuned_path,n_dimensions=pca_dimensions, num_transformer_layers=6)




transfer.compare_draw(test_actuals_normal,test_preds_normal,test_actuals_abnormal,test_preds_abnormal)  """


AD_class_label=AD_metaframe[AD_metaframe['diagnosis'].isin(['Dementia','Normal cognition','MCI'])]
AD_class_feature=AD_frame.reindex(AD_class_label.index,axis='index')


converted_diag=AD_class_label.apply(diagnosis_conversion,axis=1)
AD_class_label.insert(1,'diagnosis_label',converted_diag)
AD_class_label=AD_class_label['diagnosis_label']
feature_x,label_y=pre_processing(AD_class_feature,AD_class_label,pca_path,scalar_path)


X_train, X_test, y_train, y_test=sklearn.model_selection.train_test_split(feature_x,label_y,train_size=0.7,stratify=np.asnumpy(label_y))#58 48 188 288
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("len")

X_train=np.asnumpy(X_train)
X_test=np.asnumpy(X_test)
y_train=np.asnumpy(y_train)
y_test=np.asnumpy(y_test)


from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN,SVMSMOTE

import collections
counter = collections.Counter(y_train)
print('Before', counter)
# oversampling the train dataset using SMOTE + ENN
smenn = ADASYN()
ros=RandomOverSampler()
X_train_smenn, y_train_smenn = smenn.fit_resample (X_train, y_train)
counter = collections.Counter (y_train_smenn)
print('After', counter)

X_train_smenn_cu=np.array(X_train_smenn)
X_test_cu=np.array(X_test)
y_train_smenn_cu=np.array(y_train_smenn)
y_test_cu=np.array(y_test)


""" model = xgboost.XGBClassifier(gamma=0.05,min_child_weight=0.1,tree_method='approx',reg_alpha=0.1,reg_lambda =0.1, n_estimators=1000, max_depth=5, eta=0.03,colsample_bytree=0.3,subsample=0.7, device='cuda',sampling_method='uniform', early_stopping_rounds=300)
model.fit(X_train_smenn_cu,y_train_smenn_cu, eval_set=[(X_test_cu, y_test_cu)])
test_results = model.predict(((X_test))) 
train_results=model.predict(((X_train)))

test_results=np.asnumpy(test_results)
train_results=np.asnumpy(train_results)

y_train=np.asnumpy(y_train)
y_test=np.asnumpy(y_test) """

transfer.transfer_classify(X_train_smenn,X_test,y_train_smenn,y_test, test_size=0.3, epochs=500,n_dimensions=pca_dimensions, num_transformer_layers=6,learning_rate=1e-05, batch_size=32,pre_trained_model_path=fine_tuned_path)



""" from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
confusion_test=confusion_matrix(y_test,test_results)

# Training confusion matrix


print("train accuracy %.2f"% sklearn.metrics.accuracy_score(y_train,train_results,))
print("test accuracy%.2f" % sklearn.metrics.accuracy_score(y_test,test_results))
print(sklearn.metrics.classification_report(test_results,y_test))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_test,
                              display_labels=model.classes_)
disp.plot()
plt.show() """