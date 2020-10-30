import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def dbscan(data, k, eps, minpts):
    # -1: Noise 0:Undefined >0 : cluster number
    c = 0
    # k is the value that shows us which first k number of columns contain the attribute data
    # Create a new column Add undefined label
    labelcolumn = len(data.columns)
    data[labelcolumn] = 0

    # Create the KDTree for the algorithm
    neighbourhoodtree = KDTree(data.iloc[:, 0:k].values)

    # print(data.dtypes)
    # print(type(data.iloc[2][labelcolumn]))
    # print(pd.isnull(data.iloc[2][labelcolumn]))

    for i in data.index:
        if data.iloc[i][labelcolumn] != 0:
            continue
        neighbourhood = neighbourhoodtree.query_ball_point(data.iloc[i, 0:k], r=eps)
        # print(len(neighbourhood))
        if len(neighbourhood) < minpts:
            data._set_value(i, labelcolumn, -1)
            continue
        c = c + 1
        data._set_value(i, labelcolumn, c)
        # remove the i from neighbourhood list to create the seedset
        # print(neighbourhood)
        neighbourhood.remove(i)
        # print(neighbourhood)
        # print("&&&&&&&&&&&")
        seedset = neighbourhood
        j = 0
        while j < len(seedset):
            q = seedset[j]
            if data.iloc[q][labelcolumn] == -1:
                data._set_value(q, labelcolumn, c)
            if data.iloc[q][labelcolumn] != 0:
                j = j + 1
                continue
            data._set_value(q, labelcolumn, c)
            neighbours = neighbourhoodtree.query_ball_point(data.iloc[q, 0:k], r=eps)
            if len(neighbours) >= minpts:
                for m in neighbours:
                    seedset.append(m) if m not in seedset else seedset
            j = j + 1

    # print(":::::::::::")
    # for i in data.index:
    #     print(data.iloc[i][labelcolumn])

    return data


def main():
    data = pd.read_csv('iris.data', header=None)
    result = dbscan(data, 4, 0.485, 6)
    result.to_csv('iris.data.dbscan.result.csv', index=False, header=False)

    data2 = pd.read_csv('Synthetic_500S_99N.csv', header=None)
    result2 = dbscan(data2, 2, 25.7, 5)
    result2.to_csv('sysnthetic.data.dbscan.result.csv', index=False, header=False)


def test():
    data = pd.read_csv('iris.data', header=None)
    k = 4
    #  print(df.columns)
    # Added empty column for labels
    data[len(data.columns)] = np.nan
    dataA = pd.DataFrame(pd.np.random.rand(100, 3))
    dataA[3] = np.nan
    print(dataA.head())
    print(dataA.values)
    print("*******")
    print(dataA.iloc[0, 0:3].values)

    # tree = KDTree(dataA.values)
    # ind = tree.query_ball_point(dataA[:1], r=0.3) #returns a list if only on epoint is queried

    tree = KDTree(dataA.iloc[:, 0:3].values)
    ind = tree.query_ball_point(dataA.iloc[1, 0:3], r=0.3)  # returns a list if only on epoint is queried

    print("+++++++++++++++")
    print(ind)
    print(type(ind))
    print(len(ind))
    print("+++++++++++++++")
    print("+++++++++++++++")
    print(dataA[:1])

    # for i in ind[0]:
    #     print(dataA.iloc[i])
    #     print("--")
    #     print(np.linalg.norm(dataA[:1]-dataA.iloc[i]))

    for i in ind:
        print(dataA.iloc[i, 0:3])
        print("--")
        print(np.linalg.norm(dataA.iloc[1, 0:3] - dataA.iloc[i, 0:3]))

    tree2 = KDTree(data.iloc[:, 0:4].values)
    ind2 = tree2.query_ball_point(data.iloc[3, 0:4], r=0.3)  # returns a list if only if point is queried

    for i in ind2:
        print(data.iloc[i, 0:4])
        print("--")
        print(np.linalg.norm(data.iloc[3, 0:4] - data.iloc[i, 0:4]))


if __name__ == "__main__":
    main()
