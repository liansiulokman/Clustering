import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats
import statistics
# TODO: import other packages as necessary


# Read in data: df

# TODO: insert code here to perform the given task. Don't forget to document your code!

######### Part a - Data Preprocessing #########
#Check data structure, all column are non-null
df.info()
#Look at how data looks like
df.head()
#Check if there are NA values in the data
df.shape
print('\nFeatures:', df.columns)
df_na = (df.isna().sum()/df.shape[0])*100
print('\nNumber of NAs:\n')
print(df_na)
#Result: no NA values

#Turn data set to numpy format
X = df.to_numpy()

#Check X's shape
X.shape

#Plot data
# plt.figure()
# plt.scatter(X[:,0], X[:,2], c="black")
# plt.show(block=True)

#Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)


def printSScoreChart(h,X):
    print('Print S Score Chart ...')
    silhouette_score_values=list()
    NumberOfClusters=range(3,10)
    for i in NumberOfClusters:
        h_result = fcluster(h, i, criterion="maxclust")
        labels= h_result
        silhouette_score_values.append(silhouette_score(X,labels ,metric='euclidean', sample_size=None, random_state=None))
    plt.plot(NumberOfClusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters ")
    plt.show()


def printBoxPlots(labels,df):
    print('Printing box plots for each feature ...')
    columns = df.columns
    df['lable'] = labels
    for m, n in enumerate(columns):
        df.boxplot(column=n, by='lable', figsize = (10, 5), showfliers=False)
        plt.show()


def stats_to_df(d,cols,medians):
    tmp_df = pd.DataFrame(columns=cols)
    try:
        tmp_df.loc[0] = np.round(d.minmax[0],2)
        tmp_df.loc[1] = np.round(medians,2)
        tmp_df.loc[2] = np.round(d.mean,2)
        tmp_df.loc[3] = np.round(d.minmax[1],2)
        tmp_df.loc[4] = np.round(d.variance,2)
        tmp_df.loc[5] = np.round(d.skewness,2)
        tmp_df.loc[6] = np.round(d.kurtosis,2)
        tmp_df.index = ['Min', 'Median', 'Mean', 'Max', 'Variance', 'Skewness', 'Kurtosis']
    except Exception:
        print(d)
    return tmp_df.T


def printStats(labels,X,cols,cluster_n):
    for i in range(0,cluster_n):
        d = stats.describe(scaler.inverse_transform(X[(i+1)==labels]),axis=0)
        print('\nCluster {}:'.format(i+1), 'Number of instances: {}'.format(d.nobs), 'Percentage: {:.2f}%'.format(d.nobs/X.shape[0]*100))
        print(stats_to_df(d,cols,getMedians(labels,X,(i+1))))


def checkClusterSize(labels, clusterNumber):
    total = 0
    clusterSizes = []
    for i in range(0,clusterNumber):
        clusterCount = np.count_nonzero(labels == (i+1))
        clusterSizes.append(clusterCount)
        print("\nCluster ", (i+1), ": ", clusterCount, "  Percentage: {:.2f}%".format(clusterCount/len(labels)*100))
        total += clusterCount
    return clusterSizes


def getMedians(labels,X,cols):
    medians = []
    df = scaler.inverse_transform(X[cols==labels])
    for c in range(0,4):
        medians.append(statistics.median(df[:,c]))
    return medians


########### K-Means ###########
#Use KMeans clustering algorithm, set number of clusters = 5
#k_means = KMeans(init="k-means++", n_clusters=5, n_init=10, random_state=42)
#k_means.fit(X)

#Check result labels
#print(k_means.labels_)


########### Plot WCSS for K-Means ###########
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
#from the plot, the optimal number of clusters = 5


########### DBSCAN ###########
#Use DBSCAN clustering algorithm, set eps = 0.5 and min_samples = 3
#dbscan = DBSCAN(eps = 0.5, min_samples = 3)
#dbscan.fit(X)

#Check result labels
#dbscan_result = dbscan.labels_
#print(dbscan_result)


########### Hierarchical ###########
#Set linkage for hierarchical clustering
h = linkage(X, method="ward", metric="euclidean")

#Plot dendrogram
# plt.figure(figsize=(12, 5))
# dendrogram(h)
# plt.show(block=True)

#define clusters - from first plot, there should be 4 clusters
h_result = fcluster(h, 5, criterion="maxclust")
#print(h_result)


########### Check Silhouette Coefficient ###########
printSScoreChart(h,X)
print('S Score: {:.2f}'.format(silhouette_score(X,h_result ,metric='euclidean', sample_size=None, random_state=None)))
checkClusterSize(h_result, 5)
printStats(h_result,X,df.columns,5)
printBoxPlots(h_result,df)


########### Visualize Clusters ###########
Y = df.to_numpy()
plt.figure(figsize=(10, 7))
plt.scatter(Y[:,0], Y[:,2], c=h_result, cmap='rainbow')
plt.title('Clusters of Jewelary Customers (Hierarchical Clustering Model)')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.show()