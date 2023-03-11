import pandas as pd
import matris_train # training file 
import numpy as np

def main():

    #VARIABLES
    #bias values taken from train file 
    bias1 = matris_train.bias1 
    bias2 = matris_train.bias2

    #FUNCTIONS

    def detect_flower(out1): # detecting flower type if it is between defined values
        id = 0
        for row in out1:

            if  -1 < row < 0.7: 
                print(f"{id}. flower is Iris-setosa",row, input_ndata[id][5])

            elif 0.7 < row < 1.6:
                print(f"{id}. flower is Iris-versicolor",row, input_ndata[id][5])

            elif  1.6 < row < 5:  
                print(f"{id}. flower is Iris-virginica",row, input_ndata[id][5])

            else:
                print(f"{id}. cannot detect",row, input_ndata[id][5])

            id = id + 1
    #GET  INPUT VALUES FOR TESTING
    input_data = pd.read_csv("orn2.csv")
    input_ndata = np.array(input_data)
    input_data = input_data.iloc[:, 1:5]
    input_data = np.array(input_data)
    x = len(input_data)
    bias1_array = np.full((x, 1), bias1)
    bias2_array = np.full((x, 1), bias2)

    out1, nonreq = matris_train.ileriYay(input_data, bias1_array, bias2_array)
    detect_flower(out1)
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    main()

