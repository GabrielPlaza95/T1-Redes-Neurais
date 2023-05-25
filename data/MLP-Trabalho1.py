# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:06:25 2023

@author: ADM
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import seaborn as sns



# iris = load_iris()

# # Load data into a DataFrame
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# # Convert datatype to float
# df = df.astype(float)
# # append "target" and name it "label"
# df['label'] = iris.target
# # Use string label instead
# df['label'] = df.label.replace(dict(enumerate(iris.target_names)))

# # label -> one-hot encoding
# label = pd.get_dummies(df['label'])
# label.columns = ['label_' + str(x) for x in label.columns]
# df = pd.concat([df, label], axis=1)
# # drop old label
# df.drop(['label'], axis=1, inplace=True)

# # Creating X and y
# X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
# # Convert DataFrame into np array
# X = np.asarray(X)
# y = df[['label_setosa', 'label_versicolor', 'label_virginica']]
# # Convert DataFrame into np array
# y = np.asarray(y)




########################################################################
# BASE PIMA
# load pima indians dataset
#dataset = np.loadtxt("pima.csv", delimiter=",")
# basecsv = pd.read_csv('pima.csv',sep=';')
# dataset = basecsv[[
# 'A1',
# 'A2',
# 'A3',
# 'A4',
# 'A5',
# 'A6',
# 'A7',
# 'A8',
# 'saida']]

# # split into input (X) and output (Y) variables
# # X = dataset[:,0:8]
# # y = dataset[:,8]
# X= dataset.iloc[:,0:8]
# y= dataset.iloc[:,8]
# print("y:", y)

# encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_Y = encoder.transform(y)
# y = encoded_Y

#

#dataset import
dataset = pd.read_csv('cell_train.csv') #You need to change #directory accordingly

#sns.pairplot(dataset, hue="price_range")


print(dataset.head(10)) #Return 10 rows of data
#Changing pandas dataframe to numpy array
X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values
atributos=20

#Normalizacao
sc = StandardScaler()
X = sc.fit_transform(X)
X=sc.fit_transform(X)

#Categorizacao one-hot da saida
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.20 )

# sm = SMOTE(random_state=42)
# X_res_train, y_res_train = sm.fit_resample(X_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_res_train))
# X_train=X_res_train
# y_train=y_res_train


# nn-> número de neuronios na camada escondida
for nn in range (1,10,10):
    # Adding layer via add() method
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(atributos,)))
    model.add(Dense(4, activation='softmax'))  #softmax para mais classes activation='sigmoid'
    #model.summary()
    
    model.compile(
    optimizer='adam',#adam , sgd, rmsprop  
    loss='categorical_crossentropy', # mais categorias: categorical_crossentropy
    metrics=['accuracy']
    )
    
    
    ces = EarlyStopping(
    monitor='val_accuracy', 
    patience=10,
    min_delta=0.001, 
    mode='max'
    )
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy',  mode='max', verbose=1, save_best_only=True) #

    history = model.fit(X_train, y_train,
                    batch_size= 64,
                    epochs= 200,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[ces, mc]
                   )

    
    #%matplotlib inline
    #%config InlineBackend.figure_format = 'svg'
    def plot_metric(history, metric):
        train_metrics = history.history[metric]
        val_metrics = history.history['val_'+metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'b--')
        plt.plot(epochs, val_metrics, 'r-')
        plt.title('Training and validation '+ metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_"+metric, 'val_'+metric])
        plt.show()
        
    plot_metric(history, 'loss')
    plot_metric(history, 'accuracy') 
    
    
    print('########################################################################')
    print("Resultdos Teste:")
    y_pred = model.predict(X_test)
    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    acc=accuracy_score(y_test, y_pred,normalize=True)
    print("cm", cm)
    
    # load the saved model
    #saved_model = load_model('best_model.h5')
    # evaluate the model
    #y_pred = saved_model.predict(X_test)
