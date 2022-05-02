from utils import *
from stratified import *

def initialize_weights(ohe_category,layer_parameter, biasterm=True):
    weight_matrix_list = []

    inputcategory, outputcategory = [],[]
    inputindex, outputindex = [],[]
    n = 0
    for i in ohe_category:
        if ohe_category[i] != 'class_numerical':
            inputcategory.append(i) # name of the input category
            inputindex.append(n) # index of the input category
        else:
            outputcategory.append(i) # name of the output category  
            outputindex.append(n) # index of the output category
        n += 1
    
    b = 1 if biasterm == True else 0
    
    updatedlayerparameterwbias = [len(inputcategory)+b] + list(np.array(layer_parameter)+b) + [len(outputcategory)] # [inputlayer, layerparameters, outputlayer]
    for i in range(len(updatedlayerparameterwbias)-1):
        layernow = updatedlayerparameterwbias[i]
        layernext = updatedlayerparameterwbias[i+1]-1 if i !=len(updatedlayerparameterwbias)-2 else updatedlayerparameterwbias[i+1] 
        # ^ for the last layer, the bias is not included, so don't need to minus 1 ^
        init_weight = np.random.rand(layernext,layernow) * 2 - 1 # initialize the weight with random number between -1 and 1
        weight_matrix_list.append(init_weight)
        
    return weight_matrix_list

def costfunction(expected_output, actual_output):
    j = -np.dot(expected_output,np.log(actual_output) + np.dot((1 - expected_output),np.log(1 - actual_output)))
    return j

def sumofweights(listofweights,bias=True): # computes the square of all weights of the network and sum them up
    sum = 0
    for weight in listofweights:
        if bias:
            sum += np.sum(np.square(weight[1:]))
        else:
            sum += np.sum(np.square(weight))
    return sum