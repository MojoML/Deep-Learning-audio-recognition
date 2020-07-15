#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# This is a project for audio word recognition using a deep neural network 


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[ ]:


# --------------------- LOAD DATA ------------------------ #

features = np.load("feat.npy", allow_pickle = True)
path = np.load("path.npy", allow_pickle = True)
train = pd.read_csv("train(1).csv", delimiter = ",")
test = pd.read_csv("test(1).csv", delimiter = ",")


# In[ ]:


# ------------------- TRANSFORM DATA ---------------------- #

# create dictionary: key = path and value = feat
dic = {} 
for i in range(len(path)):
    dic[path[i]] = features[i]


# In[ ]:


# this function take as argument a pandas data frame and a dictionary
# and create a new list according to which path in the data frame is in the dictionary

def create_list(data_frame,dic):
    new_list= []
    for i in range(len(data_frame)):
        if data_frame["path"][i] in dic.keys():
            new_list.append(dic[data_frame["path"][i]])
    return new_list


def padding(data):
    zeros_list=[0,0,0,0,0,0,0,0,0,0,0,0,0]
    for example in range(len(data)):
        if data[example].shape[0]!=99:
            to_change=data[example].tolist()
            for adding in range(99-len(to_change)):
                to_change.append(zeros_list)
            data[example]=np.array(to_change)     
    return data
    


# In[ ]:


# split test and train 
training_data = create_list(train,dic)
test_data = create_list(test,dic)

# padding
training_data = padding(training_data)
test_data = padding(test_data)

# convert to array
training_data = np.array(training_data)
test_data = np.array(test_data)

#check shape
training_data.shape,test_data.shape


# In[ ]:


# ------------- set up data for running DECISION FOREST ----------------#

# encode the target
train_numpy = train.values
labels = train_numpy[:,1]
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)


# In[ ]:


## SPLITTING
X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)


# In[ ]:


# check shape
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


# ---------- features engineering for decision forest  -------#

def features_mean(signal):
    return np.mean(signal,axis=2)

def features(signal, functions):
    summaries=[]
    for fn in functions:
        summaries.append(fn(signal,axis=2))
    return np.concatenate(summaries,axis=1)

summaries = [np.mean, np.min, np.max, np.std]

X_train_summaries = features(X_train, summaries)
X_test_summaries = features(X_test, summaries)

X_train_mean = features_mean(X_train)
X_test_mean = features_mean(X_test)


# In[ ]:


## this function take as argument a training and a validation data set and return the accuracy based on 
# the number of nodes, which is encode here as n_estimators

def run_forest(n_estimators ,X_train, X_test, y_train = y_train, y_test = y_test , random_state = 333):
    
    acc=[] # list of accuracy which depends on the hyperparameter n_estimators

    for num_features in n_estimators: # hyperparameters to change with higher number
        forest = RandomForestClassifier(n_estimators=num_features, 
                               bootstrap = True,
                               max_features = 'sqrt',
                                random_state = random_state)
        fitted_model=forest.fit(X_train,y_train)
        prediction=fitted_model.predict(X_test)
        accuracy=accuracy_score(y_test,prediction)
        solution = (num_features,accuracy)
        acc.append(solution)
        
    return acc


# In[ ]:


#set up different number of nodes 
n_estimators_mean = [40,50,60,70]
n_estimators_summaries = [200,250,300]


# In[ ]:


# accuracy for the mean
accuracy_mean = run_forest(n_estimators_mean,X_train_mean,X_test_mean)
print(accuracy_mean)


# In[ ]:


# accuracy for the summaries
accuracy_summaries = run_forest(n_estimators_summaries,X_train_summaries,X_test_summaries)
print(accuracy_summaries)


# In[ ]:


# ------------------- SET UP DATA FOR CNN 2D ------------------- #


# In[ ]:


# change name of variables 
train_X = training_data
test_X = test_data
train_Y = labels


# In[ ]:


# reshape in fourd dimensions for input CNN
train_X = train_X.reshape(-1, 99,13, 1)
test_X = test_X.reshape(-1, 99,13, 1)
train_X.shape, test_X.shape


# In[ ]:


# transform data type in float
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')


# In[ ]:


# transform the labels
train_Y_one_hot = to_categorical(train_Y)


# In[ ]:


## CREATE THE VALIDATION SET 
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, 
                                                           random_state=13)


# In[ ]:


# check all the shape
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


# In[ ]:


#----------------------------------- first attempt ---------------------------------#

# set up hyperparameters 
batch_size = 64
epochs = 10
num_classes = 35 # fix
np.random.seed(222)


# In[ ]:


# set up the layers 
fashion_model = Sequential()

fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(99,13,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))

fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))  
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


fashion_model.compile(loss=keras.losses.categorical_crossentropy, 
                      optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[ ]:


## check the summary
fashion_model.summary()


# In[ ]:


## train and test the accuracy in the validation set
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,
                                  verbose=1,validation_data=(valid_X, valid_label))


# In[ ]:


# --------- try with drop out, 2nd attempt ----------- #


# In[ ]:


# set up hyperparameters 
batch_size = 124
epochs = 30
num_classes = 35 # fix
np.random.seed(222)


# In[ ]:


## set up the dropout to improve accuracy
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(99,13,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))

fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))  
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


fashion_model.compile(loss=keras.losses.categorical_crossentropy, 
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])


# In[ ]:


fashion_model.summary()


# In[ ]:


fashion_train = fashion_model.fit(train_X, train_label, 
                                  batch_size=batch_size,epochs=epochs,verbose=1,
                                  validation_data=(valid_X, valid_label))


# In[ ]:


#--------------------------- CNN 1D ---------------------------#


# In[ ]:


#go back to two three dimensions
train_X = training_data
test_X = test_data
train_Y = labels

train_X = train_X.reshape(-1, 99,13)
test_X = test_X.reshape(-1, 99,13)

# transform the labels
train_Y_one_hot = to_categorical(train_Y)

train_X.shape, test_X.shape


# In[ ]:


## CREATE THE VALIDATION SET 
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, 
                                                           random_state=13)


# In[ ]:


# set up hyperparameters 
batch_size = 256
epochs = 100
num_classes = 35 # fix
np.random.seed(222)


# In[ ]:


fashion_model = Sequential()

fashion_model.add(Conv1D(64, kernel_size=6,activation='relu',padding='same',input_shape=(99,13)))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))

fashion_model.add(Dropout(0.2))
fashion_model.add(Conv1D(128, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(Conv1D(128, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))

fashion_model.add(Conv1D(128, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))


fashion_model.add(Conv1D(128, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))

fashion_model.add(Conv1D(256, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))

fashion_model.add(Conv1D(256, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))

fashion_model.add(Conv1D(256, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))

fashion_model.add(Conv1D(512, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))

fashion_model.add(Conv1D(1024, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))

fashion_model.add(Conv1D(1024, kernel_size=6, activation='relu',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling1D(pool_size=2,padding='same'))
fashion_model.add(Dropout(0.2))

fashion_model.add(Flatten())
fashion_model.add(Dense(1024, activation='relu'))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.2))
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


fashion_model.summary()


# In[ ]:


fashion_model.compile(loss=keras.losses.categorical_crossentropy, 
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])


# In[ ]:


fashion_train = fashion_model.fit(train_X, train_label, 
                                  batch_size=batch_size,epochs=epochs,verbose=1,
                                  validation_data=(valid_X, valid_label))


# In[ ]:


# ----------------------- PREDICTION ON TEST SET ---------------------- #


# In[ ]:


#### make prediction on the test set
test_prediction = fashion_model.predict(test_X) # take the prediction from the model
test_prediction = np.argmax(np.round(test_prediction),axis=1) # take the index of the maximum probability
test_prediction = encoder.inverse_transform(test_prediction) # from number label to word
print(test_prediction)


# In[ ]:


## add the column of prediction to the test csv
test_csv = pd.read_csv("test.csv", delimiter = ",",index_col=None)
test_csv['word'] = test_prediction


# In[ ]:


## save the csv
out_csv = "result.csv"
test_csv.to_csv(out_csv,index=None,header=True)

