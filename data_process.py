
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from dummyPy import OneHotEncoder


## raw data (for read)
fp_train = "../data/train.csv"
fp_test  = "../data/test.csv"

## subsample training set
fp_sub_train_f = "../data/sub_train_f.csv"

fp_col_counts = "../data/col_counts"

## data after selecting features (LR_fun needed)
## and setting rare categories' value to 'other' (feature filtering)
fp_train_f = "../data/train_f.csv"
fp_test_f  = "../data/test_f.csv"

## storing encoder for labeling / one-hot encoding task
fp_lb_enc = "../data/lb_enc"
fp_oh_enc = "../data/oh_enc"


cols = ['C1', 
        'banner_pos', 
        'site_domain', 
        'site_id',
        'site_category',
        'app_id',
        'app_category', 
        'device_type', 
        'device_conn_type',
        'C14', 
        'C15',
        'C16']

cols_train = ['id', 'click']
cols_test  = ['id']
cols_train.extend(cols)
cols_test.extend(cols)


## data reading
df_train_ini = pd.read_csv(fp_train, nrows = 10)
df_train_org = pd.read_csv(fp_train, chunksize = 1000000, iterator = True)
df_test_org  = pd.read_csv(fp_test,  chunksize = 1000000, iterator = True)


cols_counts = {}  # the categories count for each feature
for col in cols:
    cols_counts[col] = df_train_ini[col].value_counts()


for chunk in df_train_org:
    for col in cols:
        cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())


for chunk in df_test_org:
    for col in cols:
        cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())

for col in cols:
    cols_counts[col] = cols_counts[col].groupby(cols_counts[col].index).sum()
    # sort the counts
    cols_counts[col] = cols_counts[col].sort_values(ascending=False)  


pickle.dump(cols_counts, open(fp_col_counts, 'wb'))


fig = plt.figure(figsize=(20,20))
for i, col in enumerate(cols):
    ax = fig.add_subplot(4, 3, i+1)
    ax.fill_between(np.arange(len(cols_counts[col])), cols_counts[col].get_values())
    ax.set_title(col)
plt.show()


k = 99
col_index = {}
for col in cols:
    col_index[col] = cols_counts[col][0: k].index

df_train_org = pd.read_csv(fp_tr ain, dtype = {'id': str}, chunksize = 1000000, iterator = True)
df_test_org  = pd.read_csv(fp_test,  dtype = {'id': str}, chunksize = 1000000, iterator = True)

hd_flag = True  # add column names at 1-st row
for chunk in df_train_org:
    df = chunk.copy()
    for col in cols:
        df[col] = df[col].astype('object')
        # assign all the rare variables as 'other'
        df.loc[~df[col].isin(col_index[col]), col] = 'other'
    with open(fp_train_f, 'a') as f:
        df.to_csv(f, columns = cols_train, header = hd_flag, index = False)
    hd_flag = False

hd_flag = True  # add column names at 1-st row
for chunk in df_test_org:
    df = chunk.copy()
    for col in cols:
        df[col] = df[col].astype('object')
        # assign all the rare variables as 'other'
        df.loc[~df[col].isin(col_index[col]), col] = 'other'
    with open(fp_test_f, 'a') as f:
        df.to_csv(f, columns = cols_test, header = hd_flag, index = False)      
    hd_flag = False 

## 1.label encoding
lb_enc = {}
for col in cols:
    col_index[col] = np.append(col_index[col], 'other')

for col in cols:
    lb_enc[col] = LabelEncoder()
    lb_enc[col].fit(col_index[col])

## store the label encoder
pickle.dump(lb_enc, open(fp_lb_enc, 'wb'))


## 2.one-hot encoding
oh_enc = OneHotEncoder(cols)

df_train_f = pd.read_csv(fp_train_f, index_col=None, chunksize=500000, iterator=True)
df_test_f  = pd.read_csv(fp_test_f, index_col=None, chunksize=500000, iterator=True)

for chunk in df_train_f:
    oh_enc.fit(chunk)
for chunk in df_test_f:
    oh_enc.fit(chunk)


pickle.dump(oh_enc, open(fp_oh_enc, 'wb'))

n = sum(1 for line in open(fp_train_f))   # total size of train data (about 46M)
s = 2000000 # desired train set size (2M)

## the 0-indexed header will not be included in the skip list
skip = sorted(random.sample(range(1, n), n-s-1)) 
df_train = pd.read_csv(fp_train_f, skiprows = skip)
df_train.columns = cols_train

## store the sub-sampling train set as .csv
df_train.to_csv(fp_sub_train_f, index=False) 

