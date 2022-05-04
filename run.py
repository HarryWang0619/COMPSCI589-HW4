from utils import *
from stratified import *

def importhousedata():
    house = importfile('hw3_house_votes_84.csv', ',')
    housecategory = {}
    for i in house[0]:
        housecategory[i] = 'categorical'
    housecategory["class"] = 'class'
    housedata = np.array(house[1:]).astype(float)
    return housedata, housecategory

def importwinedata():
    wine = importfile('hw3_wine.csv', '\t')
    winecategory = {}
    for i in wine[0]:
        winecategory[i] = 'numerical'
    winecategory["# class"] = 'class'
    winedata = np.array(wine[1:]).astype(float)
    return winedata, winecategory

def importcancerdata():
    cancer = importfile('hw3_cancer.csv', '\t')
    cancercategory = {}
    for i in cancer[0]:
        cancercategory[i] = 'numerical'
    cancercategory["Class"] = 'class'
    cancerdata = np.array(cancer[1:]).astype(float)
    return cancerdata, cancercategory

def importcmcdata():
    cmc = importfile('cmc.data', ',')
    cmccategory = {"Wife's age":"numerical","Wife's education":"categorical",
    "Husband's education":"categorical","Number of children ever born":"numerical",
    "Wife's religion":"binary","Wife's now working?":"binary",
    "Husband's occupation":"categorical","Standard-of-living index":"categorical",
    "Media exposure":"binary","Contraceptive method used":"class"}
    cmcdata = np.array(cmc).astype(int)
    return cmcdata, cmccategory


if __name__=="__main__":
    housedata, housecategory = importhousedata()
    winedata, winecategory = importwinedata()
    cancerdata, cancercategory = importcancerdata()
    cmcdata,cmccategory = importcmcdata()
    ohe_house_data, ohe_house_category = onehotencoder(housedata, housecategory)
    ohe_wine_data, ohe_wine_category = onehotencoder(winedata,winecategory)
    ohe_cmc_data, ohe_cmc_category = onehotencoder(cmcdata,cmccategory)
    ohe_cancer_data, ohe_cancer_category = onehotencoder(cancerdata,cancercategory)
