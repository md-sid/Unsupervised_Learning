import numpy as np
import pandas as pd
import assessment as assessment
import dbscan as dbscan
import matplotlib.pyplot as plt


def main():
    print("Running Iris data Experiments")
    parameters = [(0.1, 6), (0.15, 6), (0.2, 6), (0.3, 6), (0.485, 6), (0.7, 6), (1, 6)]
    index = 0
    for parameter in parameters:
        plt.close('all')
        data = pd.read_csv('iris.data', header=None)
        result = dbscan.dbscan(data, 4, parameter[0], parameter[1])
        result[len(result.columns)] = result.index
        clusterDetatails(result, 5, 6, parameter[0], parameter[1])
        puri = assessment.purity(result, 5, 6)
        print("purity: " + str(puri))
        sill = assessment.silhouettecofficient(result, 6, 4)
        print("Silhoutte Coeff: " + str(sill[0]))
        result = result.sort_values([5, len(result.columns)-1], ascending=(True, False))
        # result = result.sort_values(len(result.columns) - 1, ascending=False)
        result = result.reset_index(drop=True)
        # print(result.head())
        plt.figure()
        # result.plot.scatter(x=np.array(result.shape), y=len(result.columns) - 1)
        plotname = 'irisData.dbscan.sill.result.' + str(index) + '.png'
        plt.scatter(np.arange(result.shape[0]), result[len(result.columns) - 1])
        plt.savefig(plotname, dpi=300)
        index = index + 1

    print("Running Synthetic data experiments")
    synParameters = [('Synthetic_300S_63N.csv', 35.5, 4), ('Synthetic_400S_63N.csv', 25.5, 5),
                     ('Synthetic_500S_0N.csv', 23.5, 6), ('Synthetic_500S_34N.csv', 26, 5),
                     ('Synthetic_500S_66N.csv', 26, 5), ('Synthetic_500S_82N.csv', 25.7, 4),
                     ('Synthetic_500S_99N.csv', 25.7, 4), ('Synthetic_600S_99N.csv', 24.7, 6),
                     ('Synthetic_700S_179N.csv', 24.7, 6)]
    index = 0
    for parameter in synParameters:
        plt.close('all')
        data = pd.read_csv(parameter[0], header=None)
        result = dbscan.dbscan(data, 2, parameter[1], parameter[2])
        result[len(result.columns)] = result.index
        clusterDetatails(result, 3, 4, parameter[1], parameter[2])
        puri = assessment.purity(result, 3, 4)
        print(puri)
        sill = assessment.silhouettecofficient(result, 4, 2)
        print(sill[0])
        # result = result.sort_values(3)
        result = result.sort_values([3, len(result.columns) - 1], ascending=(True, False))
        # result = result.sort_values(len(result.columns) - 1, ascending=False)
        result = result.reset_index(drop=True)
        # print(result.head())
        plt.figure()
        # result.plot.scatter(x=np.array(result.shape), y=len(result.columns) - 1)
        plotname = 'synthetic.dbscan.sill.result.' + str(index) + '.png'
        plt.scatter(np.arange(result.shape[0]), result[len(result.columns) - 1])
        plt.savefig(plotname, dpi=300)
        index = index + 1


#
#
#
# ('Synthetic_300S_63N.csv',16,3),('Synthetic_400S_63N.csv',16,3),('Synthetic_500S_0N.csv',16,3),('Synthetic_500S_34N.csv',16,3),('Synthetic_500S_66N.csv',16,3),('Synthetic_500S_82N.csv',16,3) ,('Synthetic_500S_99N.csv',16,3) ,('Synthetic_600S_99N.csv',16,3) ,('Synthetic_700S_179N.csv',16,3)
def clusterDetatails(data, gtlabel, clabel, eps, minpts):
    print("radius: " + str(eps) + " min points: " + str(minpts))
    labels = data[clabel - 1].unique()
    glabels = data[gtlabel - 1].unique()
    count = 0
    noiseClusters = 0
    for l in glabels:
        if l == 0:
            noiseClusters = noiseClusters + 1
        else:
            count = count + 1
    print("Ground Truth Clusters: " + str(count) + " Noise Clusters: " + str(noiseClusters))

    count = 0
    noiseClusters = 0
    for l in labels:
        if l == -1:
            noiseClusters = noiseClusters + 1
        else:
            count = count + 1
    print("Clusters: " + str(count) + " Noise Clusters: " + str(noiseClusters))


if __name__ == '__main__':
    main()
