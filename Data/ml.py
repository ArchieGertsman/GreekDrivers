import pandas as pd
import numpy as np 
import tensorflow as tf
import keras
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Data: 
    "Constructor class for creating the training and testing vectors"
    def __init__(self, pickle_file,vectors=["lat","lon","speed","lon_acc","lat_acc","avg_speed","bearing","edge_progress","vehicle_density","avg_surr_speed"]):
        self.df = pd.read_pickle(pickle_file).reset_index()
        self.unique_ids = self.df[self.df["type"].isin(["Car","Taxi"])].id.unique()
        self.vectors = vectors
        self.data_dict = defaultdict(list)
        self.CreateVectors()
        self.PartitionData()
        self.encoder = LabelEncoder()
        self.encoder.fit(self.y_data)
        y_train = self.encoder.transform(self.y_train)
        y_test = self.encoder.transform(self.y_test)
        
    def CreateVectors(self):
        for idx in self.unique_ids:
            for vector in self.vectors:
                self.data_dict[idx].append(self.df[self.df["id"]==idx][vector])
        
    def PartitionData(self):
        self.x_data = np.array([self.data_dict[idx] for idx in self.unique_ids])
        self.y_data = np.array([self.df[self.df["id"]==idx]["type"].iloc[0] \
                        for idx in self.unique_ids])
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(\
            self.x_data,self.y_data,test_size=0.2,shuffle=False)
        
class Network:
    "Class for creating various LSTMs"
    def __init__(self,Data,neurons_lstm,neurons_dense,epochs):
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.epochs=epochs
        self.Data = Data

    def Baseline(self):
        self.model = keras.models.Sequential([
            keras.layers.LSTM(self.neurons_lstm, return_sequences=True),
            keras.layers.Dense(self.neurons_dense)            
        ])

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def FitModel(self):
        self.model.fit(self.Data.x_train.self.Data.y_train, \
            validation_data=(self.Data.x_test,self.Data.y_test), epochs=self.epochs)