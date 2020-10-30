import itertools
from itertools import chain, combinations

import numpy as np
import pandas as pd

from assessment import silhouettecofficient
import assessment as assessment

from numpy.random import rand


def k_means(df, K, epsilon):

    D = df.copy()
    t = 0                           # iteration number
    dim = len(D.columns)-1          # number of attributes
    N_t = len(D.index)              # number of samples

    mu = np.zeros(shape=(K, dim))


    # print("Min Max range of input data")
    D_min = D.min().values
    D_max = D.max().values
    # print(D_min)

    # Initailization according to K-means++
    # ref : https://www.geeksforgeeks.org/ml-k-means-algorithm/
    for k in range(K):
        if k ==0 :
            mu[k, :] = D_min[0:dim]+(D_max[0:dim] - D_min[0:dim])*rand(dim)
        else:
            min_dist = np.ones(shape=(1, N_t))*100
            for j in range(N_t):
                xj = D.iloc[j].values
                for i in range(k):
                    err_k = xj[0:dim] - mu[i, :]
                    dist = np.linalg.norm(err_k, ord=2)**2
                    min_dist[0, j] = min(min_dist[0, j], dist)
            j_max = np.argmax(min_dist)
            xj_max = D.iloc[j_max].values
            mu[k, :] = xj_max[0:dim]


    # print("Initial Mean")
    # print(mu)


    while True:
        # print("iteration")
        t = t+1
        # print(t)
        # Creating a empty Dataframe for cluster assignment
        C = pd.DataFrame(np.zeros(shape=(N_t, 1)), columns=["Kmeans_Class"], dtype=int)

        mu_old = mu.copy()

        # Cluster Assignment step
        for j in range(N_t) :
            xj = D.iloc[j].values

            res = np.zeros(shape=(1,K))
            for k in range(K):

                err_k = xj[0:dim] - mu[k, :]
                res[0, k] = np.linalg.norm(err_k, ord=2)**2

            i_star = int(np.argmin(res)+1)

            C.iloc[j] = i_star

        # Centroid update
        for k in range(K):
            idx_k = C.loc[C['Kmeans_Class']==k+1].index

            if len(idx_k) != 0:
                c_k = D.iloc[idx_k.values]
                mu[k, :] = c_k.mean()[0:dim]


        if sum(np.linalg.norm(np.subtract(mu, mu_old), ord=2, axis=1) ** 2) <= epsilon:
            # print("Number of iterations")
            # print(t)
            # print("SSE(mu)")
            # print(sum(np.linalg.norm(np.subtract(mu, mu_old), ord=2, axis=1) ** 2))
            D[dim+1] = C
            return D
            break

def K_means_w_epochs(data, K, epsilon, epochs):
    print("K =")
    print(K)
    print("epsilon =")
    print(epsilon)

    dim = len(data.columns) - 1
    sil_v = -1

    for epc in range(epochs):
        # print("Epochs =")
        # print(epc)
        out_epoch = k_means(data, K, epsilon)
        out_epoch[len(out_epoch.columns)] = out_epoch.index
        sill_temp = silhouettecofficient(out_epoch, dim + 2, dim)
        # print("silhouete")
        # print(sill_temp[0])
        if sil_v < sill_temp[0]:
            sil_v = sill_temp[0]
            sil_best_epoch = sill_temp
            result_best_epoch = out_epoch
    # print("Best silhouete")
    # print(sil_best_epoch[0])
    return result_best_epoch, sil_best_epoch

def K_means_find_parameters(data, kRange, epsRange, epochs):


    sil_metric = np.zeros([len(kRange), len(epsRange)])
    for k in range(len(kRange)):
        for e in range(len(epsRange)):
            out, sil = K_means_w_epochs(data, kRange[k], epsRange[e], epochs)
            if all(sil_metric < sil[0]):
                result_best = out
                # sil_best = sil
            sil_metric[k, e] = sil[0]
    print(sil_metric)
    return result_best




def main():
    print('**************K means*********')

    kRange = [2, 3, 4, 5, 6, 7, 8, 9]
    epsRange = [0.1]
    epochs = 3

    data = pd.read_csv('iris.data', header=None)
    result = K_means_find_parameters(data, kRange, epsRange, epochs)
    result.to_csv('iris.data.result.k_means.csv', index=False, header=False)

    # data2 = pd.read_csv('Synthetic_Data_Label.csv', header=None)
    data2 = pd.read_csv('Synthetic_500S_66N.csv', header=None)
    result2 = K_means_find_parameters(data2, kRange, epsRange, epochs)
    result2.to_csv('sysnthetic.data.result.k_means.csv', index=False, header=False)





def test():
    df = pd.read_csv('iris.data', header=None)
    print(df.describe())
    # print(df[0])

    k = 3
    epsilon = 0.001

    print("Calling K means")
    k_means(df, k, epsilon)

    print("k means done")


if __name__ == "__main__":
    main()
