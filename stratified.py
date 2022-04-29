from utils import *

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


#!!!!!!!! I will change the validation function to let it take neural network.

# def kfoldcrossvalid(data, categorydict, k=10, ntree=10, maxdepth=5, minimalsize=10, minimalgain=0.01, algortype='id3', bootstrapratio = 0.1):
#     folded = stratifiedkfold(data, categorydict, k)
#     listofnd = []
#     accuracylist = []
#     for i in range(k):
#         # print("at fold", i)
#         testdataset = folded[i]
#         foldedcopy = folded.copy()
#         foldedcopy.pop(i)
#         traindataset = np.vstack(foldedcopy) 
#         correctcount = 0
#         trainforest = plantforest(traindataset,categorydict,ntree,maxdepth,minimalsize,minimalgain,algortype,bootstrapratio)
#         emptyanalysis = []
#         # testdataset = traindataset
#         for instance in testdataset:
#             predict, correct = forestvote(trainforest,instance,categorydict)
#             emptyanalysis.append([predict, correct])
#             if predict == correct:
#                 correctcount += 1
#         listofnd.append(np.array(emptyanalysis))
#         accuracylist.append(correctcount/len(testdataset))
#     acc = np.mean(accuracylist)
#     return listofnd, acc