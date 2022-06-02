
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import math
import cv2
train_df = pd.read_csv('./labels_index.csv')
labels_map = {}
def label_f(m):
    labels_map[int(m.label_index)] = m.cultivar
train_df.apply(label_f, axis=1)

fgvc9label = train_df.cultivar.unique()

train_df2 = pd.read_csv('./train_dataset_by_image.csv')
df = pd.DataFrame()
for i in tqdm(range(len(train_df2))):
    # data_month = int(train_df2.iloc[i, 0].split('/')[-1][5:7])
    # data_day = int(train_df2.iloc[i, 0].split('/')[-1][8:9])
    data_month = int(train_df2.iloc[i, 3].split('/')[0])
    data_day = int(train_df2.iloc[i, 3].split('/')[1])
    # print(data_month, data_day)
    if (data_month ==6 and  1<=data_day<=30) and ('PI_' + train_df2.iloc[i, 2][2:]) in fgvc9label:
        tmp = pd.concat([train_df2.iloc[i:i+1, 0:1], train_df2.iloc[i:i+1, 2:3]], axis=1)
        tmp.iloc[0, 1] = 'PI_' + tmp.iloc[0, 1][2:]
        df = pd.concat([df, tmp], axis=0)

df_all = pd.DataFrame()
for i in tqdm(range(len(df))):
    image_path = df.iloc[i, 0]
    data_input = cv2.imread('./data/' + image_path)

    s_path = './fgvc8_data/data/train/'+image_path.split('/')[3][:-4]+'.jpeg'
    try:
        cv2.imwrite(s_path, data_input, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        df.iloc[i, 0] = './fgvc8_data/data/train/' + image_path.split('/')[3][:-4] + '.jpeg'
        df_all = pd.concat([df_all, df.iloc[i:i+1, :]], axis=0)
    except:
        continue

df_all.to_csv(('fgvc8_data.csv'), index=None)



print('over')