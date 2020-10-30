import numpy as np
import pandas as pd


def main():
    # data = pd.read_csv('iris.data.dbscan.result.csv', header=None)
    # data[len(data.columns)] = data.index
    # print("printing assessment metrics of iris data on dbscan")
    # puri = purity(data, 5, 6)
    # print(puri)
    # sill = silhouettecofficient(data, 6, 4)
    # print(sill)

    data2 = pd.read_csv('sysnthetic.data.dbscan.result.csv', header=None)
    data2[len(data2.columns)] = data2.index
    print("printing assessment metrics of synthetic data on dbscan")
    puri = purity(data2, 3, 4)
    print(puri)
    sill = silhouettecofficient(data2, 4, 2)
    print(sill[0])

def test():
    print("Test")
    data = pd.read_csv('iris.data.result.csv', header=None)
    unique = data[4].unique()
    print(unique)
    data[len(data.columns)] = data.index
    data.to_csv('test.scv', index=False, header=False)
    # assessment(data, 5, 6)


def purity(data, truthColumn, labelColumn):
    # create the ground truth lists

    indexColumn = len(data.columns) - 1
    groundTruthList = data.groupby(truthColumn - 1)[indexColumn].apply(list).values.tolist()
    # print(groundTruthList)
    groundTruthListCount = len(groundTruthList)
    # print(groundTruthListCount)

    # create the cluster list
    clusterList = data.groupby(labelColumn - 1)[indexColumn].apply(list).values.tolist()
    # print(clusterList)
    clusterListCount = len(groundTruthList)
    # print(clusterListCount)

    puritySum = 0
    for i in clusterList:
        iSet = set(i)
        count = 0
        for j in groundTruthList:
            jSet = set(j)
            temp = iSet & jSet
            if temp and len(temp) > count:
                count = len(temp)
        puritySum = puritySum + count

    purity = puritySum / data.shape[0]

    return purity


def silhouettecofficient(data, labelColumn, kn):
    indexColumn = len(data.columns) - 1
    clusterList = data.groupby(labelColumn - 1)[indexColumn].apply(list).values.tolist()

    uIn = np.zeros(data.shape[0])
    uOut = np.zeros(data.shape[0])
    sI = np.zeros(data.shape[0])
    for i in clusterList:
        for j in i:
            uIn[j] = calculateUin(j, i, data, kn)
            uOut[j] = calculateUout(j, i, clusterList, data, kn)

    result = 0
    for index in range(0, data.shape[0]):
        sI[index] = (uOut[index] - uIn[index]) / max(uOut[index], uIn[index])
        result = result + (uOut[index] - uIn[index]) / max(uOut[index], uIn[index])

    result = result / data.shape[0]
    data[len(data.columns)] = pd.Series(sI)
    return result, sI


def calculateUout(index, cluster, clusterList, data, kn):
    baseTuple = np.array(data.iloc[index, 0:kn])
    minDistance = np.inf
    for k in clusterList:
        sum = 0
        if clusterList.index(k) == clusterList.index(cluster):
            continue
        else:
            # calculate the distance sum to this cluster
            for i in k:
                tempTuple = np.array(data.iloc[i, 0:kn])
                sum = sum + np.linalg.norm(baseTuple - tempTuple)
        result = sum / len(k)
        if result < minDistance:
            minDistance = result

    return 1000000 if minDistance == np.inf else minDistance


def calculateUin(index, cluster, data, kn):
    baseTuple = np.array(data.iloc[index, 0:kn])
    sum = 0
    for i in cluster:
        if i == index:
            continue
        else:
            tempTuple = np.array(data.iloc[i, 0:kn])
            sum = sum + np.linalg.norm(baseTuple - tempTuple)

    result = sum / (len(cluster) - 1)
    # print(result)
    return result


if __name__ == "__main__":
    main()
