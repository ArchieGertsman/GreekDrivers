import pandas as pd
import numpy as np 
import tensorflow as tf
from collections import defaultdict

class Data: 
    "Constructor class for creating the training and testing vectors"
    def __init__(self, pickle_file,vectors=["lat","lon","speed","lon_acc","lat_acc","avg_speed","bearing","edge_progress","vehicle_density","avg_surr_speed"]):
        self.df = pd.read_pickle(pickle_file).reset_index()
        self.unique_ids = self.df[self.df["type"].isin(["Car","Taxi"])].id.unique()
        self.vectors = vectors
        self.data_dict = defaultdict(list)

    def CreateVectors(self):
        for idx in self.unique_ids:
            for vector in self.vectors:
                self.data_dict[idx].append(df[df["id"]==idx][vector])

