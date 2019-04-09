#!/usr/bin/env python3

import sys
import numpy as np
from tensorflow.keras.models import load_model

def get_sepsis_score(values, column_names):
    Empty_M=np.zeros((66,1));
    feature_select=np.vstack((0,1,2,3,4,5,6,7,21,35,39));
    l=len(values)
    M0=values;
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
    model = load_model('my_model_v6.h5')
    labels= model.predict_classes(X)
    scores= model.predict_proba(X)
    scores=scores[:,1];
    return (scores, labels)


def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    return (values, column_names)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: %s input[.psv]' % sys.argv[0])

    record_name = sys.argv[1]
    if record_name.endswith('.psv'):
        record_name = record_name[:-4]

    # read input data
    input_file = record_name + '.psv'
    (values, column_names) = read_challenge_data(input_file)

    # generate predictions
    (scores, labels) = get_sepsis_score(values, column_names)

    # write predictions to output file
    output_file = record_name + '.out'
    with open(output_file, 'w') as f:
        count=0;
        for (s, l) in zip(scores, labels):
#            f.write('%g|%d\n' % (s, l))
            count=count+1;
            if count==len(scores):
             f.write('%g|%d' % (s, l))
            else:
             f.write('%g|%d\n' % (s, l))
                
