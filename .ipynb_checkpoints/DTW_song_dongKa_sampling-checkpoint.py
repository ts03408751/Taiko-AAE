
# coding: utf-8

import glob as glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


def Main_Execure(Dir):
    
    
    Train_data = Input_motifs(Dir)
    distances = np.zeros((np.shape(Train_data)[0],np.shape(Train_data)[0]))
    print(Train_data.shape)
    print(Train_data.shape[0])
    print(Train_data.shape[1])

    w = Train_data.shape[1]
    for ind,i in enumerate(Train_data):
        for c_ind,j in enumerate(Train_data):
            cur_dist = 0.0
            #Find sum of distances along each dimension
            for z in range(np.shape(Train_data)[2]):
                cur_dist += DTWDistance(i[:,z],j[:,z],w)
            distances[ind,c_ind] = cur_dist

    clusters, curr_medoids = cluster(distances, 5)

    #挑出分類群中最多的那群資料
    array_whichMany = []
    for i in range(len(np.unique(clusters))):
        index_i = np.unique(clusters)[i]
        index_i = (np.where(clusters==index_i)[0])
        locals()['samples_%s'%i] = Train_data[index_i]

        array_whichMany.append(len(locals()['samples_%s'%i]))

    which_group = array_whichMany.index(max(array_whichMany))
    samples = locals()['samples_%s'%(which_group)]
    
    timeseries_sample = []
    for i in range(samples.shape[1]):
        axe_mean = []

        for j in range(samples.shape[2]):
            axis_total =(samples[:,i,j])
            mean = axis_total.mean()  #sample shape= (20,7) 的 第i個time的第j個軸的平均值
            axe_mean.append(mean) #從軸0~7依序的平均值

        timeseries_sample.append(axe_mean) #20個timeseries都完成

    timeseries_sample = np.array(timeseries_sample)
    timeseries_sample = timeseries_sample.reshape(1,timeseries_sample.shape[0],timeseries_sample.shape[1])  ##最後的合成訊號
    
    return timeseries_sample


def preprocess(df):
        
        df = np.asarray(df, dtype=np.float32)
        if len(df.shape) == 1:
            raise ValueError('Data must be a 2-D array')

    #     if np.any(sum(np.isnan(df)) != 0):
    #         print('Data contains null values. Will be replaced with 0')
    #         df = np.nan_to_num()

        #standardize data 
        df = StandardScaler().fit_transform(df)
        #normalize data
        df = MinMaxScaler().fit_transform(df)
        return df


def Input_motifs(Dir):
    path = Dir+'/*/*.csv'
    filenames = glob.glob(path)

    samples_notScaling = []

    for filenames_ in filenames:
        samples_notScaling.append(pd.read_csv(filenames_))

    samples_Scaling=[]  

    for df in samples_notScaling:
        sample_No = preprocess(df)
        samples_Scaling.append(sample_No)
        
    samples_Scaling = np.array(samples_Scaling)
    
    return samples_Scaling

# DTW
import sys
import random
import matplotlib.pyplot as plt
import math as m
import importlib
from mpl_toolkits.mplot3d import Axes3D

def cluster(distances, k):

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k) 
    new_medoids = np.array([-1]*k)

    # To be repeated until mediods stop updating
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # Update cluster medoids to be lowest cost point. 
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        print('Mediods still not equal')

    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)

def DTWDistance(s1,s2,w):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])
 
def LB_Keogh(s1,s2,r):
    '''
    Calculates LB_Keough lower bound to dynamic time warping. Linear
    complexity compared to quadratic complexity of dtw.
    '''
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return np.sqrt(LB_sum)


# sample = Main_Execure('D:/Ming/motif/aaaaa/song1/order1/don')
