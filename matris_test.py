import pandas as pd
import matris_train
import numpy as np

bias1 = 0.1

#FUNCTIONS

def activ_func(inp):
        if inp < 0:
            return 0.01 * inp
        else:
            return inp

def detect_flower(id, out1):
    
    if  -1 < out1 < 0.7: 
        print(f"{id}. flower is Iris-setosa",out1, input_ndata[id][5])

    elif 0.7 < out1 < 1.4:
        print(f"{id}. flower is Iris-versicolor",out1, input_ndata[id][5])

    elif  1.4 < out1 < 5:  
        print(f"{id}. flower is Iris-virginica",out1, input_ndata[id][5])

    else:
        print(f"{id}. cannot detect",out1, input_ndata[id][5])

def test(input_matris, weight_data):
    
    hid = input_matris.dot(weight_data)
    out1 = hid + bias1
    out1 = activ_func(out1)
    return out1
    

#GET  INPUT VALUES FOR TESTING
input_data = pd.read_csv("orn2.csv")
input_ndata = np.array(input_data)
input_data = input_data.iloc[:, 1:5]
input_data = np.array(input_data)

#GET REQUIRED WEIGHTS
weight_data = pd.read_csv("weights.csv")
weight_data = np.array(weight_data)
weight_data = weight_data[:,1]

id = 0
for row in input_data:
    input_matris = row
    out1 = test(input_matris, weight_data)
    bias1 = matris_train.bias1
    detect_flower(id, out1)
    id = id + 1   
    


