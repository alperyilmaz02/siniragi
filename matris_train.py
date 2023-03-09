import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

#initial values
#give weights and bias random values between 0.5 and 0.6
w1 = random.uniform(0.5,0.6)
w2 = random.uniform(0.5,0.6)
w3 = random.uniform(0.5,0.6)
w4 = random.uniform(0.5,0.6)
w5 = random.uniform(0.5,0.6)
w6 = random.uniform(0.5,0.6)
w7 = random.uniform(0.5,0.6)
w8 = random.uniform(0.5,0.6)
w9 = random.uniform(0.5,0.6)
w10 = random.uniform(0.5,0.6)
w11 = random.uniform(0.5,0.6)
w12 = random.uniform(0.5,0.6)
w13 = random.uniform(0.5,0.6)
w14 = random.uniform(0.5,0.6)
w15 = random.uniform(0.5,0.6)
bias1 = random.uniform(0.5,0.6)
bias2 = random.uniform(0.5,0.6)

w1_array = np.array([[w1],[w2],[w3],[w4]])
w2_array = np.array([[w5],[w6],[w7],[w8]])
w3_array = np.array([[w9],[w10],[w11],[w12]])
wo_array = np.array([[w13],[w14],[w15]])

# CONSTANTS
LR = np.array([0.0001])
LR_array = np.full((4,1),LR)
exp_out1 = np.array([[0]])  #Iris-setosa
exp_out2 = np.array([[1]])  #Iris-versicolor
exp_out3 = np.array([[2]])  #Iris-virginia



def test_train():
    dataFrame = pd.read_csv("Iris.csv", index_col ="Id")
    dataFrame = np.array(dataFrame)

    train, test = train_test_split(dataFrame, test_size=0.2)

    df = pd.DataFrame(train)
    df.to_csv("orn1.csv")

    df = pd.DataFrame(test)
    df.to_csv("orn2.csv")

def activ_func( inp):
    if inp < 0:
        return 0.01 * inp
    else:
        return inp

def hatahesaplama( out_matris, dataFrame):
    for row in dataFrame:
        id = 0
        flower = row[5]

        if flower == "Iris-setosa":
            exp_out = exp_out1
        elif flower == "Iris-versicolor":
            exp_out = exp_out2
        elif flower == "Iris-virginica":
            exp_out = exp_out3

        err = 1/2 * pow((out_matris[id] - exp_out),2)
        id = id + 1

    return err

def ileriYay( newDataFrame, bias1_array, bias2_array):

    #hidden layer 1
    hid = newDataFrame.dot(w1_array)
    hid1 = hid + bias1_array
    
    x = len(hid1)
    id = 0
    matris1 = np.full((x, 1), None)
    matris2 = np.full((x, 1), None)
    matris3 = np.full((x, 1), None)
    matris = np.full((x, 1), None)
   
    for row in hid1:
        matris1[id] = float(activ_func(row))
        id = id + 1
    
    #hidden layer 2
    id = 0
    hid = newDataFrame.dot(w2_array)
    hid2 = hid + bias1_array
    
    for row in hid2:
        matris2[id] = float(activ_func(row))
        id = id + 1

    #hidden layer 3
    id = 0
    hid = newDataFrame.dot(w3_array)
    hid3 = hid + bias1_array
    
    for row in hid3:
        matris3[id] = float(activ_func(row))
        id = id + 1
    
    #out layer
    id = 0
    #combining matrices
    com_matris = np.hstack((matris1, matris2, matris3))
    
    out = com_matris.dot(wo_array)
    out1 = out + bias2_array
   
    for row in out1:
        for num in row:
            matris[id] = float(activ_func(num))
        id = id + 1
    
    return matris, com_matris

def geriYay(out_matris, com_matris, dataFrame, w1_array, w2_array, w3_array, wo_array, newDataFrame, bias1, bias2):
    
    id = 0
    for row in dataFrame:

        flower = row[5]

        if flower == "Iris-setosa":
            exp_out = exp_out1
        elif flower == "Iris-versicolor":
            exp_out = exp_out2
        elif flower == "Iris-virginica":
            exp_out = exp_out3

        if out_matris[id] >= 0:
           
            turev1 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w13
            w1_array = w1_array - np.transpose(LR * turev1)
    
            turev2 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w14
            w2_array = w2_array - np.transpose(LR * turev2)

            turev3 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w15
            w3_array = w3_array - np.transpose(LR * turev3)
            
            turevout = (out_matris[id] - exp_out) * com_matris[id]
            wo_array = wo_array - np.transpose(LR * turevout)
            
        elif out_matris[id] < 0:
            turev1 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w13
            w1_array = w1_array - np.transpose(LR * turev1) * 0.01 * 0.01

            turev2 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w14
            w2_array = w2_array - np.transpose(LR * turev2) * 0.01 * 0.01

            turev3 = (out_matris[id] - exp_out).dot([newDataFrame[id]]) * w15
            w3_array = w3_array - np.transpose(LR * turev3) * 0.01 * 0.01

            turevout = (out_matris[id] - exp_out) * com_matris[id] * 0.01
            wo_array = wo_array - np.transpose(LR * turevout)

        id = id + 1
        
    return w1_array, w2_array, w3_array, wo_array, bias1, bias2

test_train() #split dataset as test(%20) and train(%80) 
err_array = [] #create an array for error values

#INPUT
dataFrame = pd.read_csv("orn1.csv")
dataFrame = np.array(dataFrame)
newDataFrame = dataFrame[:, 1:5]
x = len(dataFrame)
itr = 1

# train dataset for 1000 iterations
while(itr < 1000):
    bias1_array = np.full((x, 1), bias1)
    bias2_array = np.full((x, 1), bias2)
    out_matris, com_matris = ileriYay(newDataFrame, bias1_array, bias2_array)
    err = hatahesaplama(out_matris, dataFrame)
    err_array = np.append(err_array, err)
    w1_array, w2_array, w3_array, wo_array, bias1, bias2 = geriYay(out_matris, com_matris, dataFrame, w1_array, w2_array, w3_array, wo_array, newDataFrame, bias1, bias2)
    itr = itr + 1
    
#show error
err_array = np.array(err_array)
plt.plot(err_array)
plt.ylabel('error')
plt.show()

# open the file in the write mode
with open('weights.csv', 'w') as f:
    #create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(w1_array)
    writer.writerow(w2_array)
    writer.writerow(w3_array)
    writer.writerow(wo_array)





