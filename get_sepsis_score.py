#!/usr/bin/env python3

import sys
import numpy as np
from tensorflow.keras.models import load_model

def get_sepsis_score(data):
    Empty_M=np.zeros((66,1));
    feature_select=np.vstack((0,1,2,3,4,5,6,7,21,35,39));
    l=len(data)
    M0=data;
    X=np.zeros((l,121));    
    for j in range(0,len(M0)):
         if j<5:
             temp=Empty_M
             for k in range(0,j+1):
                 temp[11*k:11*k+11]=M0[k,feature_select];
         else:
              temp=Empty_M
              for k in range(j-5,j+1):
                temp[11*(k-j+5):11*(k-j+5)+11]=M0[k,feature_select];       
         temp =  np.vstack((temp, (temp[0:11]-temp[11:22])))
         temp =  np.vstack((temp, (temp[11:22]-temp[22:33])))
         temp =  np.vstack((temp, (temp[22:33]-temp[33:44])))
         temp =  np.vstack((temp, (temp[33:44]-temp[44:55])))
         temp =  np.vstack((temp, (temp[44:55]-temp[55:66])))
         X[j,:]=np.transpose(temp);
    X=np.nan_to_num(X)
    X = X.reshape(len(X),11,11,1);
    model = load_model('my_model_v8.h5')
    labels= model.predict_classes(X)
    scores= model.predict_proba(X)
    scores=scores[:,1];
    return (scores, labels)


def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

if __name__ == '__main__':
    # read data
    data = read_challenge_data(sys.argv[1])

    # make predictions
    if data.size != 0:
        (scores, labels) = get_sepsis_score(data)

    # write results
    with open(sys.argv[2], 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        if data.size != 0:
            for (s, l) in zip(scores, labels):
                f.write('%g|%d\n' % (s, l))


