
import numpy as np
import pickle, gzip
import MLP as mlp
import os
import cv2
import matplotlib.pyplot as plt

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

# Creating the MLP object initialize the weights
mlp_classifier = mlp.Mlp(size_layers = [256, 30, 20, 12], 
                         act_funct   = 'relu',
                         reg_lambda  = 0,
                         bias_flag   = True)



train_class_list = []
hist_data__train_list = []
train_dataset_path = "/my_dataset/train/"
for train_class in os.listdir(train_dataset_path):
    for train_data in os.listdir(train_dataset_path + train_class):
        img_data = cv2.imread(train_dataset_path + train_class + '/'+ train_data,0)
        hist,bins = np.histogram(img_data.ravel(),256,[0,256])
        train_class_list.append(train_class)
        hist_data__train_list.append(hist)



# Training data
train_X = hist_data__train_list
train_y = train_class_list
# change y [1D] to Y [2D] sparse array coding class
n_examples = len(train_y)
labels = np.unique(train_y)
train_Y = np.zeros((n_examples, len(labels)))
for ix_label in range(len(labels)):
    ix_tmp = np.where(train_y == labels[ix_label])[0]
    train_Y[ix_tmp, ix_label] = 1


test_class_list = []
hist_data_test_list = []
test_dataset_path = "/my_dataset/test/"
for test_class in os.listdir(test_dataset_path):
    for test_data in os.listdir(test_dataset_path + test_class):
        img_data = cv2.imread(test_dataset_path + test_class + '/'+ test_data,0)
        hist,bins = np.histogram(img_data.ravel(),256,[0,256])
        test_class_list.append(test_class)
        hist_data_test_list.append(hist)


# Test data
test_X = hist_data_test_list
test_y = test_class_list
# change y [1D] to Y [2D] sparse array coding class
n_examples = len(test_y)
labels = np.unique(test_y)
test_Y = np.zeros((n_examples, len(labels)))
for ix_label in range(len(labels)):
    ix_tmp = np.where(test_y == labels[ix_label])[0]
    test_Y[ix_tmp, ix_label] = 1


# Training with Backpropagation and 400 iterations
iterations = 400
str_train_y = ''
list_int_train_y = []
str_y_hat = ''
list_int_y_hat = []
loss = zerolistmaker(iterations)
for ix in range(iterations):
        mlp_classifier.train(train_X, train_Y, 1)
        Y_hat = mlp_classifier.predict(train_X)
        y_tmp = np.argmax(Y_hat, axis=1)
        y_hat = labels[y_tmp]
        for i in range(len(train_y)):
                str_train_y = str_train_y + train_y[i] + ','

        for q in range(len(str_train_y)):
                if str_train_y[q] == ',':
                        continue
                else:
                        list_int_train_y.append(int(str_train_y[q]))

        for i in range(len(y_hat)):
                str_y_hat = str_y_hat + y_hat[i] + ','

        for q in range(len(str_y_hat)):
                if str_y_hat[q] == ',':
                        continue
                else:
                        list_int_y_hat.append(int(str_y_hat[q]))

        for n in range(1199):
                loss.append((0.5)*np.square(list_int_y_hat[n ] - list_int_train_y[n ]).mean())








# Ploting loss vs iterations
plt.figure()
ix = np.arange(iterations)
#print("loss  = " , loss )
print(len(loss))

acc = np.mean(1 * (y_hat == train_y))
print('Training Accuracy: ' + str(acc*1000))
