import sys
sys.path.append('../../Lib/')
import pandas as pd
import numpy as np
from feature_eng import split_trajectories
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time
from sklearn.preprocessing import PolynomialFeatures
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import seaborn as sns

def speed_ratio(grp, min_speed=0):
    return len(grp[grp.speed > min_speed]) / len(grp)

def split_train_test(df,test_size):
    #dataframe is split into training and test set such that the number of cars and taxis in test set is the same
    
    df_id = df.reset_index()[["file_name",'id','type']].drop_duplicates()
    X,y = df_id[["file_name","id"]],df_id['type']
    X_train,X_test,_,y_test = train_test_split(X, y, test_size=test_size, random_state=4, stratify=y) 
    
    df_train = df[df.index.droplevel(['time','edge_id']).isin(X_train.set_index(['file_name','id']).index)]
    X_test['type'] = y_test
    g = X_test.groupby('type')
    
    #ensures that number of cars and taxis in test set are equal
    X_test = g.apply(lambda group: group.sample(g.size().min())).reset_index(drop = True)
    df_test = df[df.index.droplevel(['time','edge_id']).isin(X_test.set_index(['file_name','id']).index)]
    return df_train,df_test

def basic_accuracy(X,y, model):
# find f1_score and accuracy
    y_hat = model.predict(X)
    return __accuracy(y,y_hat)

def voting_accuracy(X,y, model,aggregate_by = 'id',predict_proba = False,display = False):
# 
    if predict_proba == True:
        
        y_hat = pd.DataFrame(index = y.index,data = model.predict_proba(X),columns = model.classes_)
        y_hat_orig = y_hat.copy()

        if aggregate_by == 'id':
            
            #predicted value for the entire trajectory would be the mode of the predicted labels
            y_hat = y_hat.groupby(['file_name','id']).mean()
            y_test = y.groupby(['file_name','id']).first(['type'])
             
        elif aggregate_by == 'edge':

            #predicted value for the entire trajectory would be the mode of the predicted labels
            y_hat = y_hat.groupby(['file_name','id','edge_id']).mean()
            y_test = y.groupby(['file_name','id','edge_id']).first(['type'])
        
        y_hat = y_hat.idxmax(axis=1)
        
    else:

        y_hat = pd.DataFrame(index = y.index,data =  model.predict(X),columns = ['type'])
        y_hat_orig = y_hat.copy()
        y_hat_orig['type'] = (y_hat_orig['type'] == 'Car').astype(int)
        
        if aggregate_by == 'id':
            
            #predicted value for the entire trajectory would be the mode of the predicted labels
            y_hat = y_hat.groupby(['file_name','id']).apply(lambda group: pd.Series.mode(group['type'])[0])
            y_test = y.groupby(['file_name','id']).first(['type'])
            
        elif aggregate_by == 'edge':
            
            #predicted value for the entire trajectory would be the mode of the predicted labels
            y_hat = y_hat.groupby(['file_name','id','edge_id']).apply(lambda group: pd.Series.mode(group['type'])[0])
            y_test = y.groupby(['file_name','id','edge_id']).first(['type'])

    if display:
        __display_voting_members(y_hat_orig,y_hat,y_test,X)
        
    return __accuracy(y_test,y_hat)

 

def get_xy(df,overlap = None,traj_len = None,agg_dict = None,min_movement_limit = 0.75,outlier_limit=None,balance = None,downsample_feature_list = None,col_factor = None,window = None):
    """get X and y from dataframe using either aggregation or downsampling"""
    
    
    if agg_dict is not None:
        """split id into smaller trajectories of length = traj_len and aggregate it by statistical parameters outlined in agg_dict"""
        df_agg =__rolling_agg(df, window_size=traj_len, step=int((1 - overlap)*traj_len),agg_dict = agg_dict)
        df_agg = df_agg[df_agg.speed_bool_count*min_movement_limit <= df_agg.speed_bool_sum]
        df_agg.drop(['speed_bool_count','speed_bool_sum'],inplace= True,axis = 1)
       
    elif downsample_feature_list is not None:
        """downsample the id into smaller trajectories by splitting it into trajectories of length = window_size, taking its mean  
    and pivoting it to create downsampled trajectories of length col_factor"""
        
        df_agg = __downsample(df,downsample_feature_list,col_factor,window)
      
    if outlier_limit is not None:
        """remove data outside the outlier_limit"""
        df_agg = __filter_by_percentile(df_agg,outlier_limit)

    if balance == 'by_edge':
        """ the number of cars and taxis in each edge is balanced"""
        df_agg['type_count'] = df_agg['type']
        g_count = df_agg.groupby(['edge_id','type'], group_keys=False).count()['type_count']
        g = df_agg.groupby(['type','edge_id'], group_keys=False)
        df_agg = g.apply(lambda grp: grp.sample(min(g_count.loc[(grp.index.get_level_values(2)[0],slice(None))])))
        df_agg.drop('type_count',inplace = True,axis = 1)

    if balance == 'by_type':
        """ number of cars and taxis are balanced"""
        g = df_agg.groupby('type', group_keys=False)
        df_agg = g.apply(lambda grp: grp.sample(g.size().min()))

    X,y = df_agg.drop('type', axis=1), df_agg.type
    return X,y
  

   
class voting_model():
    """ voting model fits a ML model on trajectory probabilities in id to improve accuracy """
    def __init__(self,model,X,y):
        self.model = model
        self.voting_model = self.fit(X,y)
             
    def fit(self,X,y):
        """fit quadratic weighted function on model output using X,y"""
        
        X_log = self.generate_op_df(X)
        Y_log = y.groupby(['file_name','id']).first(['type']).apply(lambda x: 1 if (self.model.classes_[0] == x) else -1 )
        model = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
        
        return model.fit(X_log,Y_log)
    
    def generate_op_df(self,X):
        
        model_output = pd.DataFrame(data = self.model.predict_proba(X)[:,0],index = X.index,columns = ['x_1'])
        df_agg  = model_output.groupby(['file_name','id']).agg({'x_1': ['mean','std','count','max','min']})
        df_agg.columns = ['_'.join(col) for col in df_agg.columns]
        df_agg.fillna(0,inplace = True)
        
        return df_agg
    
    def predict(self,X):
        
        X_test = self.generate_op_df(X)
        model_output = self.voting_model.predict(X_test)
        model_output = np.vectorize(lambda x: self.model.classes_[0] if (x>=0) else self.model.classes_[1])(model_output)
        
        return model_output
    
    def accuracy(self,X,y):
        
        y_test = y.groupby(['file_name','id']).first(['type'])
        y_hat = self.predict(X)
        y_hat = pd.DataFrame(index = y_test.index,data = y_hat,columns = ['type'])
        
        a = y_hat['type']==y_test
   
        f = f1_score((y_test == 'Car').astype(int),(y_hat == 'Car').astype(int))
        return len(a[a==True]) / len(y_test),f

        
class ensemble():
    """ creates ensemble of models by considering best "model_num" number of models """
    def __init__(self,model_num,accuracy_measure,model_list = None):
        self.model_num = model_num
        self.accuracy_measure = accuracy_measure
        self.model_list = model_list
        
    def find_ensemble(self,df_acc,traj_len,vehicle_density,predict_proba = False):
        self.is_predict_proba = predict_proba
        self.model_list = df_acc.loc[(slice(None),'accuracy','mean'),(vehicle_density,traj_len,self.accuracy_measure)].sort_values(ascending = False).index.get_level_values(0)[:self.model_num].to_list()
      
    def fit(self,X,y,model_dict=None):
        self.model_dict = model_dict
        
        if model_dict == None:
            self.model_dict = {}
            for model in self.model_list:
                self.model_dict[model] = model.fit(X,y)
        
        #get model.classes_ from first model in the dictionary
        values_view = model_dict.values()
        value_iterator = iter(values_view)
        self.classes_ = next(value_iterator).classes_  
                
    def predict(self,X):
        label_list = []
        df_model = pd.DataFrame(columns = self.model_list)
        
        if self.is_predict_proba == False:
            for model in self.model_list:
                df_model[model] = self.model_dict[model].predict(X)
            return df_model.apply(lambda x : x.mode(),axis = 1)[0].to_numpy()
            
        else:
            return self.predict_proba(X,get_label = True)
    
    def predict_proba(self,X,get_label = False):
        
        label_list = []
        model = list(self.model_dict.values())[0]
        df_model = pd.DataFrame(columns = pd.MultiIndex.from_product([self.model_list,model.classes_]))#,index = np.arange(0,len(X)))

        for name in self.model_list:
            model = self.model_dict[name]
            
            df_model.loc[:,(name,model.classes_)] = model.predict_proba(X)
            
        df_model = df_model.mean(axis=1, level=[1])
        
        if get_label == True:
            return df_model.idxmax(axis=1).to_numpy()
        else:
            return df_model.to_numpy()
 
# some helper functions
  
def __filter_by_percentile(df,percentile):
    # remove top and bottom 'percentile' of data from dataframe
    top_le = 1-(percentile/100)
    bottom_le = percentile/100
    df_top = df.quantile(top_le).reset_index()
    df_top['cond'] ='('+df_top['index']+" <= "+df_top[top_le].astype(str)+')'
    df_bottom = df.quantile(bottom_le).reset_index()
    df_bottom['cond'] ='('+df_bottom['index']+" >= "+df_bottom[bottom_le].astype(str)+')'
    df = df.query(df_top.cond.str.cat(sep=' & '))
    df = df.query(df_bottom.cond.str.cat(sep=' & '))
    
    return df 

def __rolling_agg(df, agg_dict, window_size=100, step=25):
    # rolling agg with step size = 1
    df_agg = df.groupby(df.index.names[:-1]) \
                .rolling(window_size) \
                .agg(agg_dict) \
                .dropna()

    # select a subset of above computations to achieve custom step size
    df_agg = df_agg.groupby(df_agg.index.names, 
                            as_index=False, 
                            group_keys=False) \
                .apply(lambda x: x[::step])
    
    df_agg.columns = ['_'.join(col) for col in df_agg.columns]
    
    # add 'type' column
    vehicle_types = df.type.groupby(df.index.names[:-1]).first()

    return df_agg.join(vehicle_types)

def __downsample(df,feature_list,col_factor,window):
    df_lane_len = df[['len','lanes','type']].droplevel(3).reset_index().drop_duplicates().set_index(df.index.names[:-1])
    df = df[feature_list] \
        .groupby(df[feature_list].index.names[:-1]) \
        .apply(lambda grp: __split_id_for_pivot(grp,window,col_factor)) \
        .dropna() \
        .reset_index(level=-1, drop=True)

    df.columns = [feature+'_'+str(i) for i in range(col_factor) for feature in feature_list]
    df = df.join(df_lane_len)
    
    return df


def __accuracy(y,y_hat):
    
    a = y_hat==y
    f = f1_score((y == 'Car').astype(int),(y_hat == 'Car').astype(int))
    return len(a[a==True]) / len(y),f


def __xtrack_dist_diff(df):
    """splits a vehicle trajectory into smaller trajectories of fixed size and removes
    the last (len(df) mod size) rows
    """

    df['xtrack_diff'] = df.xtrack_dist \
    .groupby(df.index.names[-1]) \
    .apply(lambda x: (x - x.shift(-1)).fillna(0))
    
    return df

def __pivot(A, col_factor):
    
    c = A.shape[1]
    if A.size < col_factor*c:
        return None
    r_new = A.size // (col_factor*c)
    A = A[:col_factor*r_new]
    return A.to_numpy().reshape(r_new, col_factor*c)

def __split_id_for_pivot(grp,window,col_factor):
    
    grp = grp.reset_index(level=(0,1,2), drop=True)
    grp.index = pd.TimedeltaIndex(grp.index,unit='s')
    grp = grp.resample(window).mean().reset_index(drop=True)
    return pd.DataFrame(__pivot(grp,col_factor))

def __display_voting_members(y_hat_orig,y_hat,y_test,X):
    
    y_hat_orig['id_traj'] = list(range(len(y_hat_orig)))
    x_plot_num = 5
    y_plot_num = int(sum(y_hat!=y_test)/x_plot_num) +1
    fig, axes = plt.subplots(y_plot_num,x_plot_num, sharey = True, figsize=(5*x_plot_num,5*(y_plot_num)))
    axes = axes.ravel()
    i = 0

    for file_name,idx in X.index.droplevel((2)).unique():
        if str(y_hat.loc[(file_name,idx)]) == str(y_test.loc[(file_name,idx)]):
            continue

        axes[i].set(ylim=(0,1))
        type_str = "predicted: "+str(y_hat.loc[(file_name,idx)]) + ", actual: "+str(y_test.loc[(file_name,idx)])
        sns.barplot(y = 'Car',x = 'id_traj',data = y_hat_orig.loc[(file_name,idx)],ax = axes[i]).set_title("file_name: "+str(file_name)+", id "+str(idx)+" \n "+type_str)
        i+=1

    fig.tight_layout(h_pad=2)   





    
    
    