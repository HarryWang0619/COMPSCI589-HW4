from sqlite3 import Row
from evaluationmatrix import *
from sklearn import datasets
import sklearn.model_selection
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

def same(attributecolumn):
    return all(item == attributecolumn[0] for item in attributecolumn)

def majority(attributecolumn):
    return np.argmax(np.bincount(attributecolumn.astype(int)))
