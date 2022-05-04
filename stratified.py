from utils import *
from run import *
from neuralnetwork import *
from evaluationmatrix import *

# In this file, I reused the stratified cross-validation method from the last assignment.

# Stratified K-Fold method
def stratifiedkfold(data, categorydict, k = 10):
    classindex = list(categorydict.values()).index("class")
    datacopy = np.copy(data).T
    classes = list(Counter(datacopy[classindex]).keys())
    nclass = len(classes) # number of classes
    listofclasses = []

    for oneclass in classes:
        index = [idx for idx, element in enumerate(datacopy[classindex]) if element == oneclass]
        oneclassdata = np.array(datacopy.T[index])
        np.random.shuffle(oneclassdata)
        listofclasses.append(oneclassdata)

    splitted = [np.array_split(i, k) for i in listofclasses]
    nclass = len(classes)
    combined = []

    for j in range(k):
        ithterm = []
        for i in range(nclass):
            if len(ithterm) == 0:
                ithterm = splitted[i][j]
            else:
                ithterm = np.append(ithterm,splitted[i][j],0)
        combined.append(ithterm)
    
    return combined

def kfoldcrossvalidneuralnetwork(raw_data, rawcategory, layerparameter, k = 10, minibatchk = 15, lambda_reg = 0.15, learning_rate = 0.01, epsilon_0 = 0.00001, softstop = 6000, printq = False):
    folded = stratifiedkfold(raw_data, rawcategory, k)
    listofnd = []
    accuracylist = []
    listofjlist = []
    for i in range(k):
        if printq:
            print('fold',i+1)
        rawtestdataset = folded[i]
        rawfoldedcopy = folded.copy()
        rawfoldedcopy.pop(i)
        rawtraindataset = np.vstack(rawfoldedcopy)
        ohe_traindata,ohe_category = onehotencoder(rawtraindataset, rawcategory)
        ohe_testdata = onehotencoder(rawtestdataset, rawcategory)[0]
        n_ohe_train,minmax = normalizetrain(ohe_traindata, ohe_category)
        n_ohe_test = normalizealltest(ohe_testdata, ohe_category, minmax)
        finalweight, jlist = train_neural_network(n_ohe_train, ohe_category, layerparameter, minibatchk, lambda_reg, learning_rate, epsilon_0, softstop, printq)
        predictvsexpect, singleaccuracy = predict_many_nn(n_ohe_test, ohe_category, finalweight)
        listofnd.append(predictvsexpect)
        accuracylist.append(singleaccuracy)
        listofjlist.append(jlist)
    acc = np.mean(accuracylist)
    return listofnd, acc, listofjlist
    