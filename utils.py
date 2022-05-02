from sqlite3 import Row
from evaluationmatrix import *
from sklearn import datasets
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
import random
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from collections import Counter

def importfile(name:str,delimit:str):
    # importfile('hw3_wine.csv', '\t')
    file = open("datasets/"+name, encoding='utf-8-sig')
    reader = csv.reader(file, delimiter=delimit)
    dataset = []
    for row in reader:
        dataset.append(row)
    file.close()
    return dataset

def onehotencoder(data,category):
    dataT=data.T.copy()
    enc = OneHotEncoder(sparse=False)
    i = 0
    appendeddict = {}
    for cat in category:
        if category[cat] == 'categorical':
            hotneeded = dataT[i]
            hotted = enc.fit_transform(hotneeded.reshape(-1,1))
            for j in enc.categories_[0]:
                newname = cat+'_'+str(j)
                appendeddict[newname] = 'ohe_numerical'
            dataT = np.append(dataT,hotted.T,axis=0)
        if category[cat] == 'class':
            hotneeded = dataT[i]
            hotted = enc.fit_transform(hotneeded.reshape(-1,1))
            for class_ in enc.categories_[0]:
                newname = cat+'_'+str(class_)
                appendeddict[newname] = 'class_numerical'
            dataT = np.append(dataT,hotted.T,axis=0)
        i += 1
    
    category.update(appendeddict)
    i = 0
    categorycopy = category.copy()
    droplist = []
    for cat in category:
        if category[cat] == 'categorical' or category[cat] == 'class':
            droplist.append(i)
            categorycopy.pop(cat)
        i += 1
    dataT = np.delete(dataT,droplist,axis=0)
    return dataT.T, categorycopy

def normalizetrain(data, category): # input data in by row/by instance
    dataTC = data.T.copy()
    minmaxes = []
    i = 0
    for oneattribute in category:
        if category[oneattribute] == 'numerical':
            colmin = np.min(dataTC[i])
            colmax = np.max(dataTC[i])
            singleminmax = [colmin,colmax]
            # normalize to 0 to 1
            for j in range(len(dataTC[i])):
                dataTC[i][j] = (dataTC[i][j] - colmin)/(colmax - colmin)
            minmaxes.append(singleminmax)
        else:
            minmaxes.append([0.0,1.0])
        i=i+1
    return dataTC.T, category, minmaxes
    

def same(attributecolumn):
    return all(item == attributecolumn[0] for item in attributecolumn)

def majority(attributecolumn):
    return np.argmax(np.bincount(attributecolumn.astype(int)))
