import numpy as np
import csv
import re
from itertools import islice
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Sigmoid函数，S型曲线，
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Sigmoid函数的导函数
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return 2.0 / (1.0 + np.exp(-2*x))-1

def tanh_derivative(x):
    return 1 - tanh(x)*tanh(x)

def relu(x):
    for index, value in enumerate(x):
        if value < 0:
            x[index] = 0
        if value >= 0:
            x[index] = x[index]
        return x

def relu_derivative(x):
    for index, value in enumerate(x):
        if value < 0:
            x[index] = 0
        if value >= 0:
            x[index] = 1
        return x

def NeuralNetwork_train(val_y,batch,sizes, learning_rate ,my_train_matrix,train_y):
    train_row = my_train_matrix.shape[0]
    #hidden_list = np.ones((train_row , sizes[1]))
    result_y = [1] * train_row
    weights_input_to_hidden = 10 * np.random.randn(sizes[1],sizes[0])
    weights_hidden_to_output = 10 * np.random.randn(sizes[1]+1)
    deta_weights_input_to_hidden = np.zeros((sizes[1], sizes[0]))
    deta_weights_hidden_to_output = [0] * (sizes[1]+1)
    hidden_matrix = np.ones((train_row, sizes[1]+1))
    sig_der_list = [1] * (sizes[1]+1)
    for k in range(0,10):
        print(k)
        count = 0

        for (result,train,my_train,hidden_list) in zip(result_y,train_y,my_train_matrix,hidden_matrix):

            for j in range(0,sizes[1]):
                hidden_list[j+1] = tanh(np.dot(my_train,weights_input_to_hidden[j]))
            for dj in range(0, sizes[1]):
                sig_der_list[dj+1] = tanh_derivative(np.dot(my_train,weights_input_to_hidden[dj]))

            z = np.dot(hidden_list, weights_hidden_to_output)
            result = np.dot(hidden_list, weights_hidden_to_output)
            deta_0 = (train - result)#* tanh_derivative(z)
            deta_weights_hidden_to_output += deta_0 * hidden_list
            deta = [1] * sizes[1]
            for kk in range(0,sizes[1]):
                deta[kk] = deta_0 * weights_hidden_to_output[kk+1] * sig_der_list[kk+1]
            for i in range(0, sizes[1]):
                for j in range(0, sizes[0]):
                    deta_weights_input_to_hidden[i][j] += deta[i] * my_train[j]
            if count % batch == 0:
                tmp1 = weights_hidden_to_output
                tmp2 = weights_input_to_hidden
                weights_hidden_to_output += (learning_rate)/train_row * deta_weights_hidden_to_output
                weights_input_to_hidden += (learning_rate)/train_row * deta_weights_input_to_hidden
                deta_weights_input_to_hidden = np.zeros((sizes[1], sizes[0]))
                deta_weights_hidden_to_output = [0] * (sizes[1]+1)
            count += 1
    #print(de_MSE)
    return weights_input_to_hidden,weights_hidden_to_output

def NeuralNetwork_test(my_val_matrix,weights_input_to_hidden,weights_hidden_to_output):
    val_row = my_val_matrix.shape[0]
    result_y = [1] * val_row
    for i in range(0,val_row):
        hidden_list = tanh(np.dot(my_val_matrix[i], weights_input_to_hidden.transpose()))
        hidden_list = np.append([1], hidden_list)
        result_y[i] = np.dot(hidden_list, weights_hidden_to_output)
    return result_y

def my_train(batch,my_train_matrix,train_row,train_col,yin,learning_rate):
    insert_one = [1]* train_row
    train_y = my_train_matrix[:,-1]
    val_y = train_y
    my_train_matrix = np.delete(my_train_matrix,train_col-1,axis = 1)
    my_train_matrix = scaling_zs(my_train_matrix)
    my_train_matrix = np.column_stack((insert_one, my_train_matrix))
    weights_input_to_hidden, weights_hidden_to_output = NeuralNetwork_train(val_y,batch,[train_col,yin,1],learning_rate ,my_train_matrix,train_y)
    return weights_input_to_hidden, weights_hidden_to_output

def my_val(val_y,my_val_matrix, val_row, val_col, weights_input_to_hidden, weights_hidden_to_output):
    MSE = 0.0
    insert_one = [1]* val_row
    my_val_matrix = np.delete(my_val_matrix,val_col-1,axis = 1)
    my_val_matrix = scaling_zs(my_val_matrix)
    my_val_matrix = np.column_stack((insert_one, my_val_matrix))
    result_y = NeuralNetwork_test(my_val_matrix,weights_input_to_hidden,weights_hidden_to_output)
    for (y1,y2) in zip(result_y,val_y):
        MSE += (1/(2*val_row))*pow( y1-y2 , 2 )
    #print(MSE)
    return MSE,result_y,val_y

def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item]

def scaling_mm(dataset):
    data_row = dataset.shape[0]
    data_col = dataset.shape[1]
    de_set = np.zeros((3,data_col))
    de_set[0] = dataset.min(0)
    de_set[1] = dataset.max(0)
    de_set[2] = de_set[1] - de_set[0]
    for i in find_all_index(de_set[2], 0):
        de_set[2][i] = 1
    for i in range(0,data_row):
        dataset[i] = (dataset[i] - de_set[0]) / de_set[2]
    return dataset

def scaling_zs(dataset):
    data_mean = np.mean(dataset, axis=0)
    data_std = np.std(dataset, axis=0)
    for i in find_all_index(data_std, 0):
        data_std[i] = 1
    for i in range(dataset.shape[0]):
        dataset[i] = np.true_divide((dataset[i]-data_mean), data_std)
    return dataset

def csv_handle(route):
    input_file = open(route)
    count = 0
    hr = []
    weathersit = []
    temp = []
    atemp = []
    hum = []
    windspeed = []
    cnt = []
    for line in islice(input_file, 1, None):
        de_instant,de_dteday,de_hr,de_weathersit,de_temp,de_atemp,de_hum,de_windspeed,de_cnt,de_n = re.split(r",|\n",line)
        #de_hr = float(de_hr)
        #de_weathersit = float(de_weathersit)
        #de_temp = float(de_temp)
        #de_atemp = float(de_atemp)
        #de_hum = float(de_hum)
        #de_windspeed = float(de_windspeed)
        #de_cnt = float(de_cnt)
        if(de_hr==""):
            print("HAHAHAHA")
            hr.append(hr[-1])
        else:
            hr.append(de_hr)
        if(de_weathersit==""):
            print("HAHAHAHA")
            weathersit.append(weathersit[-1])
        else:
            weathersit.append(de_weathersit)
        if(de_temp==""):
            print("HAHAHAHA")
            temp.append(temp[-1])
        else:
            temp.append(de_temp)
        if(de_atemp==""):
            print("HAHAHAHA")
            atemp.append(atemp[-1])
        else:
            atemp.append(de_atemp)
        if(de_hum==""):
            print("HAHAHAHA")
            hum.append(hum[-1])
        else:
            hum.append(de_hum)
        if(de_windspeed==""):
            print("HAHAHAHA")
            windspeed.append(windspeed[-1])
        else:
            windspeed.append(de_windspeed)
        if(de_cnt==""):
            print("HAHAHAHA")
            cnt.append(cnt[-1])
        else:
            cnt.append(de_cnt)
        count += 1
    row_num = len(hr)
    de_matrix = np.ones((row_num,1))
    de_matrix = np.column_stack((de_matrix, hr))
    de_matrix = np.column_stack((de_matrix, weathersit))
    de_matrix = np.column_stack((de_matrix, temp))
    de_matrix = np.column_stack((de_matrix, atemp))
    de_matrix = np.column_stack((de_matrix, hum))
    de_matrix = np.column_stack((de_matrix, windspeed))
    de_matrix = np.column_stack((de_matrix, cnt))
    de_matrix = np.delete(de_matrix, 0, axis=1)
    for row in de_matrix:
        if "?" in row:
            weizhi = np.where(row == "?")
            if(type(row[weizhi[0]]) == type("?")):
                print("HHHHA")
            row[weizhi[0]] = float(1.0)

    de_matrix = de_matrix.astype(float)
    return de_matrix


if __name__ == '__main__':
    start_time = time.time()
    train_set = 'E:/B,B,B,BBox/大三上/人工智能/Proj/regression/train.csv'
    val_set = 'E:/B,B,B,BBox/大三上/人工智能/Proj/regression/val.csv'
    test_set = 'E:/B,B,B,BBox/大三上/人工智能/Proj/regression/test.csv'

    my_train_matrix = csv_handle(train_set)
    print(my_train_matrix)
    train_row = my_train_matrix.shape[0]
    train_col = my_train_matrix.shape[1]
    print("train_row",train_row)
    print("train_col", train_col)
    bili = 3 / 4
    learning_rate = 0.5
    batch = 1
    yin = int(train_col* bili )
    weights_input_to_hidden, weights_hidden_to_output=my_train(batch,my_train_matrix, train_row, train_col,yin,learning_rate)

    my_val_matrix = csv_handle(val_set)
    val_row = my_val_matrix.shape[0]
    val_col = my_val_matrix.shape[1]
    val_y = my_val_matrix[:, -1]

    MSE, result_y, val_y = my_val(val_y,my_val_matrix, val_row, val_col,weights_input_to_hidden,weights_hidden_to_output)
    print(MSE)
    #l = len(result_y)

    #for i in range(0,l):
    #    print(result_y[i])

    end_time = time.time()
    print(end_time - start_time)