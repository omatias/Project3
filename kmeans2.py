# create initial centroid values *
# measure distance from each point to each centroid *
# Euclidean distance measure for k = # of class labels, x2, x3 *
# Manhatten distance measure for k = # of class labels, x2, x3 *
# keep track of distance measures
# assign the point to its closest centroid value
# recompute centroid value to the mean of all the points assigned to it
# restart process
from prettytable import PrettyTable
import pandas as pd
import numpy as np
# import PrettyTable
# import tabulate
from collections import Counter
import math as math
import statistics
from IPython.display import display
import random
import scipy.stats
import scipy.spatial.distance as dist
from jedi.refactoring import inline
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.physics.quantum.circuitplot import matplotlib
#%matplotlibinline
# reading in the file, creating dataframe object
file = pd.read_csv("AllGenes.csv", header=0)


def kmeans(file):
    df = pd.DataFrame(file)

    # locating the class labels and how many
    class_labels = df.iloc[:, -1].unique()
    num_labels = class_labels.size

    # subtracting class label column
    new_df = df.iloc[:, :-1]
    # print(new_df)
    # creating different k values based on amount of labels
    k = [num_labels, num_labels * 2, num_labels * 3]
    # k = [num_labels]
    rows, columns = new_df.shape
    Start = 0
    Stop = len(new_df)

    a = 0
    # iterating over k values
    e_totalWSS = 0
    e_totalBSS = 0
    m_totalWSS = 0
    m_totalBSS = 0
    m_meanBSS = 0
    e_meanWSS = 0
    e_wss_arr = []
    e_bss_arr = []
    m_wss_arr = []
    m_bss_arr = []
    for r in range(10):
        print("\nIteration: ", r, "\n")
        for i in k:
            # print("K value: ",i)
            # print("\n")
            randCentroids = []
            randNums = []
            tmp = random.randint(Start, Stop - 1)
            for g in range(i):
                while tmp in randNums:
                    tmp = random.randint(Start, Stop - 1)
                randNums.append(tmp)
            # creating random numbers for centroids
            # randNums = [np.random.randint(Start,Stop) for iter in range(Stop)]
            # print(randNums)
            # randNums = random.shuffle(randNums)
            # print(randNums)
            # value = i
            # print(value)
            # randNums = randNums[:value]
            # random.sample(range(100), 10)
            # randNums = [0,1]
            # creating random centroids
            # print("Random Numbers: ",randNums)
            for x in randNums:
                randCentroids.append(new_df.iloc[x])
            # randCentroids = np.array(randCentroids)
            # print("Initial random centroid values:")
            # print(randCentroids)

            # keeping iterations within 100
            for it in range(100):
                e_dist_vals = []
                # m_dist_vals = []
                newCentroids = []
                # gets the distance value of each point to each centroid
                for q in range(rows):
                    e_distToCentroids = []
                    # m_distToCentroids = []
                    # print(len(randCentroids))
                    for p in range(len(randCentroids)):
                        # print(p)
                        # print("row: ",q)
                        euclidean = dist.euclidean(randCentroids[p], new_df.iloc[q])
                        # print(randCentroids[p])
                        # print(new_df.iloc[q])
                        e_distToCentroids.append(euclidean)
                        # manhatten= dist.cityblock(randCentroids[p],new_df.iloc[q])
                        # print(manhatten)
                        # m_distToCentroids.append(manhatten)
                    e_dist_vals.append(e_distToCentroids)
                    # m_dist_vals.append(m_distToCentroids)
                # print("Euclidean distance values to centroids: ")
                # print(e_dist_vals)
                # print("\n")
                # print("Manhatten distance values to centroids: ")
                # print(m_dist_vals)
                # print("\n")
                e_clusters = []
                # m_clusters =[]
                # finding the centroid for each point
                for l in range(len(e_dist_vals)):
                    e_minPosition = e_dist_vals[l].index(min(e_dist_vals[l]))
                    e_clusters.append(e_minPosition)
                    # m_minPosition = m_dist_vals[l].index(min(m_dist_vals[l]))
                    # m_clusters.append(m_minPosition)
                # print("Manhatten Cluster min positions")
                # print(m_clusters)
                # print("\n")
                # print("Euclidean Cluster min positions")
                # print(e_clusters)
                # print("\n")
                # totalWss = 0
                # totalBss = 0

                # range(i) is the current k value
                # point is k
                newCentroids = []

                WSS = 0
                BSS = 0
                for point in range(0, i):

                    arr = []
                    # arr is the points belonging to each cluster
                    # e_clusters is an array of the centroid index each point belongs to

                    for row in range(rows):
                        if (e_clusters[row] == point):
                            arr.append(new_df.iloc[row])
                        # print(arr)
                        # print("array length:")
                        # print(len(arr))
                    newCentroids.append(np.mean(arr, axis=0))
                    avgOfAll = np.mean(new_df, axis=0)
                    # print("Averageg of all points: ", avgOfAll)
                    # print(newCentroids)
                    if (len(arr) != 0):
                        for h in range(len(arr)):
                            # print("Point belonging to cluster: ",arr[h])
                            # print("new centroid val: ",newCentroids[point])
                            WSS += dist.cityblock(arr[h], newCentroids[point]) ** 2
                        BSS += (dist.cityblock(avgOfAll, newCentroids[point]) ** 2) * len(arr)
                # print("Centroid index: " ,e_clusters)
                # print("Euclidean: iteration: " +str(it)+"\n" + "\nWSS: ", WSS,"\nBSS: ",BSS)
                # print("\n")
                # e_totalWSS +=WSS
                # e_totalBSS += BSS
                if (np.array_equal(newCentroids, randCentroids)):
                    e_wss_arr.append(WSS)
                    e_bss_arr.append(BSS)
                    # print("\nCentroids converged.\n")
                    break

                randCentroids = newCentroids

            # e_meanWSS = e_totalWSS/10
            # e_meanBSS = e_totalBSS/10
            # print("\nMean WSS: ",e_meanWSS)
            # print("Mean BSS: ",e_meanBSS,"\n")

            # entropyParent = ent(df.iloc[:,-1])

            # print("Parent entropy, Euclidean: ",entropyParent)

            # weightedChildAvg = entropy *
            # find mean of entire data set
            # manhatten of each centroid to that overall mean, square it
            # multiply by the number in the cluster

            # info gain - entropy of entire data set, entropy of the dataset after clustering, weighting of each cluster
            # entropy(parent-[weighted avg]* entropy[children]]

            # print("New centroids for k = ",i)
            # print(newCentroids)

            # print(randCentroids)

            # manhatten distance measures
            for it in range(100):
                m_dist_vals = []
                # gets the distance value of each point to each centroid
                for q in range(rows):
                    # e_distToCentroids = []
                    m_distToCentroids = []
                    for p in range(len(randCentroids)):
                        # euclidean= dist.euclidean(randCentroids[p],new_df.iloc[q])
                        # print(randCentroids[p])
                        # print(new_df.iloc[q])
                        # e_distToCentroids.append(euclidean)
                        manhatten = dist.cityblock(randCentroids[p], new_df.iloc[q])
                        # print(manhatten)
                        m_distToCentroids.append(manhatten)
                    # e_dist_vals.append(e_distToCentroids)
                    m_dist_vals.append(m_distToCentroids)
                # print("Euclidean distance values to centroids: ")
                # print(e_dist_vals)
                # print("\n")
                # print("Manhatten distance values to centroids: ")
                # print(m_dist_vals)
                # print("\n")
                # e_clusters = []
                m_clusters = []
                # finding the centroid for each point
                for l in range(len(m_dist_vals)):
                    m_minPosition = m_dist_vals[l].index(min(m_dist_vals[l]))
                    m_clusters.append(m_minPosition)
                    # m_minPosition = m_dist_vals[l].index(min(m_dist_vals[l]))
                    # m_clusters.append(m_minPosition)
                # print("Manhatten Cluster min positions")
                # print(m_clusters)
                # print("\n")
                # print("Euclidean Cluster min positions")
                # print(e_clusters)
                # print("\n")
                # totalWss = 0
                # totalBss = 0

                # range(i) is the current k value
                # point is k
                newCentroids = []

                WSS = 0
                BSS = 0
                for point in range(0, i):

                    arr = []
                    # arr is the points belonging to each cluster
                    # e_clusters is an array of the centroid index each point belongs to

                    for row in range(0, rows):
                        if (m_clusters[row] == point):
                            arr.append(new_df.iloc[row])
                        # print(arr)
                        # print("array length:")
                        # print(len(arr))
                    newCentroids.append(np.mean(arr, axis=0))
                    avgOfAll = np.mean(new_df, axis=0)
                    # print("Averageg of all points: ", avgOfAll)
                    # print(newCentroids)
                    if (len(arr) != 0):
                        for h in range(len(arr)):
                            # print("Point belonging to cluster: ",arr[h])
                            # print("new centroid val: ",newCentroids[point])
                            WSS += dist.cityblock(arr[h], newCentroids[point]) ** 2
                        BSS += (dist.cityblock(avgOfAll, newCentroids[point]) ** 2) * len(arr)
                # print("Centroid index: " ,e_clusters)
                # print("Manhatten: iteration: " +str(it)+"\n" + "\nWSS: ", WSS,"\nBSS: ",BSS)
                # print("\n")
                m_totalWSS += WSS
                m_totalBSS += BSS

                if (np.array_equal(newCentroids, randCentroids)):
                    # print("\nCentroids converged.\n")
                    m_wss_arr.append(WSS)
                    m_bss_arr.append(BSS)
                    break

                randCentroids = newCentroids
    print("Euclidean WSS array: ", e_wss_arr, "\n", len(e_wss_arr))
    print("Euclidean BSS array: ", e_bss_arr, "\n", len(e_bss_arr))
    print("Manhatten WSS array: ", m_wss_arr, "\n", len(m_wss_arr))
    print("Manhatten BSS array: ", m_bss_arr, "\n", len(m_bss_arr))
    arr_measures = []
    arr_measures.append(e_wss_arr)
    arr_measures.append(e_bss_arr)
    arr_measures.append(m_wss_arr)
    arr_measures.append(m_bss_arr)
    list = []
    for i in range(len(arr_measures)):
        k1 = 0
        k2 = 0
        k3 = 0
        for x in range(0, len(e_wss_arr), 3):
            k1 += arr_measures[i][x]
            k2 += arr_measures[i][x + 1]
            k3 += arr_measures[i][x + 2]
        k1 = k1 / 10
        list.append(k1)
        k2 = k2 / 10
        list.append(k2)
        k3 = k3 / 10
        list.append(k3)

        print(k1, k2, k3)
    # print(measure + " for class " + label)
    #display(arr_measures)
    wsstable = PrettyTable()

    print(k)
    map(str, k)
    wsstable.field_names = ["", k[0], k[1], k[2]]
    wsstable.add_row(["Euclidean: WSS", list[0], list[1], list[2]])
    #table.add_row(["Euclidean: BSS", list[3], list[4], list[5]])
    wsstable.add_row(["Manhatten: WSS", list[6], list[7], list[8]])
    #table.add_row(["Manhatten: BSS", list[9], list[10], list[11]])


    print(list[3], list[0])
    ratiotable = PrettyTable()
    ratiotable.field_names = ["", k[0], k[1], k[2]]
    ratiotable.add_row(["Euclidean: Ratio", (list[3]/list[0]), (list[4]/list[1]), (list[5]/list[2])])
    ratiotable.add_row(["Manhatten: Ratio", (list[9] / list[6]), (list[10] / list[7]), (list[11] / list[8])])


    print(wsstable)
    print(ratiotable)



    # m_meanWSS = m_totalWSS/10
    # m_meanBSS = m_totalBSS/10
    # print("\nMean WSS: ",m_meanWSS)
    # print("Mean BSS: ",m_meanBSS,"\n")
    # entropyParent = ent(df.iloc[:,-1])

    # print("Parent entropy, Manhatten: ",entropyParent)

    # weightedChildAvg = entropy *
    # find mean of entire data set
    # manhatten of each centroid to that overall mean, square it
    # multiply by the number in the cluster

    # info gain - entropy of entire data set, entropy of the dataset after clustering, weighting of each cluster
    # entropy(parent-[weighted avg]* entropy[children]]

    # print("New centroids for k = ",i)
    # print(newCentroids)

    # print(randCentroids)

    # for each in randCentroids:
    # total += ent(each)*(len(each)/len(randCentroids))
    # answer = entropyParent - total
    # print(newCentroids)
    # print(arr)
    # fig,ax = plt.subplots(1)
    # plt.scatter(new)
    # np.mean(arr, axis = 0))

    # calculate the mean of all the points for each centroid
    # set the new centroid values as that
    # run it again, repeat til its the same splits 3 times in a row


def ent(data):
    p_data = data.value_counts()
    entropy = scipy.stats.entropy(p_data, base=2)
    return entropy

# def infoGain(data):

kmeans(file)