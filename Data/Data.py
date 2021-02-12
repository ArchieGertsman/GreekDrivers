import numpy as np
import pandas as pd

class Data:
    "Constructor Class for creating multiindex data now and creating training and testing data later for chosen Nerual Network architecture later."
    def __init__(self,csv_file):
        self.H = 4 #header length 
        self.D = 6 #data length
        self.idx_names = ['id', 'time']
        self.col_names = ['lat', 'lon', 'speed', 'lon_acc', 'lat_acc']
        self.process_csv()
        self.frames = [self.process(self.rows[i]) for i in range(len(self.rows))]
        self.df = self.create_df()

    def process_csv(self):
        "pre processing function to turn csv file into usable material for `process`"
        in_fname = 'sample_larger.csv'

        with open(in_fname, "r") as f:
            temp = f.readlines()
    
        self.rows = temp[1:]

    def process(self,row_str):
        "Creates multi index table using 'track_id' and 'time' as the indexes. Only creates table for each index."
        parts = row_str.strip().strip(";").split(";")
        header = parts[:self.H]
        data = np.array(parts[self.H:], dtype=np.float)
        data = data.reshape(-1, self.D)

        # create MultiIndex from id and time
        timesteps = data[:,-1]
        id_arr = np.full(timesteps.shape, int(header[0].strip()))
        tups = list(zip(id_arr, timesteps))
        mul = pd.MultiIndex.from_tuples(tups, names=self.idx_names)

        data = data[:,:-1] # exclude time from data
        df = pd.DataFrame(data, columns=self.col_names, index=mul)
        df = df.assign(
            type=header[1].strip(),
            traveled_d=float(header[2]),
            avg_speed=float(header[3])
        )
        return df

    def create_df(self):
        "Concatenates all multi index tables to create onle large table."
        df = pd.concat(self.frames)
        return df

  

