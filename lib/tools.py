from __future__ import division

import itertools, operator, random
from collections import defaultdict
import csv
import pandas as pd
import numpy as np

import os

def read_all_files(dst):
    ''' get all files from one dst
    '''
    fn = []
    fn = os.listdir(dst)
    fn.remove('.DS_Store')
    return fn


def shuffle_file(content):
    ''' mix the annotated files for training/dev/test sets

    '''

    groups = [list(group) for _, group in itertools.groupby(content, operator.itemgetter(0))]
    random.shuffle(groups)
    shuffled = [line for file in groups for line in file]

    return shuffled

def analysis(data_fp, meeting, label_fp = None):
    '''select one action (rate hike/no rate hike) only if there are at least 2 votes

    '''
    df_full = []
    for i in xrange(5):
        data_in = data_fp.replace('<meeting>', meeting + '_v' + str(i))
        df = pd.read_csv(data_in, header = 0)
        if i != 0:
            drop_cols = ['Date', 'Content']
            df.drop(drop_cols, axis = 1, inplace = True)
            # if no rate hike, -1; rate hke, 1; no idea, 0
            df = df.replace(2, -1)
            df = pd.merge(df, df_full, on = ['GUID'], how = 'inner')
        df_full = df

    df_full['ML_Label'] = 0
    for i in xrange(5):
        tag = 'ML_Label_v' + str(i)
        df_full['ML_Label'] += df_full[tag]
    df_full.loc[np.abs(df_full['ML_Label']) ==1, 'ML_Label'] = 0
    df_full['ML_Label'] = np.sign(df_full['ML_Label'])


    if label_fp != None:
        label_temp = label_fp.replace('<meeting>', meeting)
        df_label = pd.read_csv(label_temp, header = 0)
        drop_cols = ['Date', 'Content']
        df_label.drop(drop_cols, axis = 1, inplace = True)
        df_label = df_label.fillna(0)
        # if no rate hike, -1; rate hke, 1; no idea, 0
        df_label = df_label.replace(2, -1)
        df_full = pd.merge(df_label, df_full, on = ['GUID'], how = 'inner')
        # performance
        correct = np.equal(df_full['ML_Label'], df_full['Annotation'])
        accuracy = np.sum(correct) / len(correct)
        recall_h = correct * np.array(df_full['Annotation'] == 1)
        recall_nh = correct * np.array(df_full['Annotation'] == -1)
        recall_np = correct * np.array(df_full['Annotation'] == 0)
        recall_h = np.sum(recall_h) / np.sum(df_full['Annotation'] == 1)
        recall_nh = np.sum(recall_nh) / np.sum(df_full['Annotation'] == -1)
        recall_np = np.sum(recall_np) / np.sum(df_full['Annotation'] == 0)
        precision_h = correct * np.array(df_full['ML_Label'] == 1)
        precision_nh = correct * np.array(df_full['ML_Label'] == -1)
        precision_np = correct * np.array(df_full['ML_Label'] == 0)
        precision_h = np.sum(precision_h) / np.sum(df_full['ML_Label'] == 1)
        precision_nh = np.sum(precision_nh) / np.sum(df_full['ML_Label'] == -1)
        precision_np = np.sum(precision_np) / np.sum(df_full['ML_Label'] == 0)
        print "Meeting %s \n Accuracy is  %.2f \n Recall: a rate hike -- %.2f, no rate hike -- %.2f, no opinion --  %.2f. \n Precision: rate hike --  %.2f, no rate hike --  %.2f, no opinion -- %.2f. \n\n" % (meeting, accuracy, recall_h, recall_nh, recall_np, precision_h, precision_nh, precision_np)

    data_out = data_fp.replace('<meeting>', meeting + '_summary')
    if label_fp != None:
        col = ['GUID', 'Date', 'Content', 'Annotation', 'ML_Label']
    else:
        col = ['GUID', 'Date', 'Content', 'ML_Label']
    df_full['GUID'] = df_full['GUID'].apply(str)
    df_full['GUID'] = '\t' +  df_full['GUID']
    df_full[col].to_csv(data_out, index = False)
