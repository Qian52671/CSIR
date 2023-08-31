from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
from keras.models import Model
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt



# model_pathinsaver = ''

k_values = list(range(3, 10))

def d_x(autoencoder,encoder,x):
    feature_model = Model(inputs=autoencoder.input, outputs=encoder.output)
    features = feature_model.predict(x)
    print(features.shape)
    features = np.reshape(features, newshape=(features.shape[0], -1))
    d_x = features
    return d_x



def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def inter_cluster_distance(x, y): 
    inter = np.ones([len(x), len(y)])
    for a in range(len(x)):
        for b in range(len(y)):
            inter[a, b] = euclidean_distance(x[a], y[b])
    return np.min(inter)

def intra_cluster_distance(x): # distance within the same cluster
    intra = np.zeros([len(x), len(x)])
    for a in range(len(x)):
        for b in range(len(x)):
            intra[a, b] = euclidean_distance(x[a], x[b])
    return np.max(intra)

def dunn_index(insert_list):
    inter = np.ones([len(insert_list), len(insert_list)])
    intra = np.zeros([len(insert_list), 1])
    clus_range = list(range(len(insert_list)))
    for a in clus_range:
        for b in (clus_range[0:a] + clus_range[a+1:]):
            inter[a, b] = inter_cluster_distance(insert_list[a], insert_list[b])
            intra[a] = intra_cluster_distance(insert_list[a])
    DI = np.min(inter) / np.max(intra)
    return DI
 

def DI_k(d_x,k_values):
    dunn_values = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=24)
        kmeans.fit(d_x)
        pred = kmeans.labels_
        dunn_df = pd.DataFrame(d_x, columns=[f'feature_{i+1}' for i in range(d_x.shape[1])])
        dunn_df['cluster'] = pred
        cluster_lists = [dunn_df[dunn_df['cluster'] == i].iloc[:, :-1].values for i in range(k)]
        DI = dunn_index(cluster_lists)
        dunn_values.append(DI)

    top_indices = np.argsort(dunn_values)[-5:]
    top_k_values = [k_values[i] for i in top_indices]
    top_dunn_values = [dunn_values[i] for i in top_indices]
    plt.figure(figsize=(10, 6))
    plt.scatter(top_k_values, top_dunn_values, color='red', label='Top 5', zorder=5)
    plt.title('Dunn Index for Different Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Dunn Index')
    plt.grid(True)
    plt.show()



def DBI_k(d_x,k_values):
    db_scores = []
    for k in k_values:
        kmeans_model = KMeans(n_clusters=k, random_state=24).fit(d_x)
        labels = kmeans_model.labels_
        score = metrics.davies_bouldin_score(d_x, labels)
        db_scores.append(score)
    
    top_indices = np.argsort(db_scores)[:5]
    top_k_values = [k_values[i] for i in top_indices]
    top_db_scores = [db_scores[i] for i in top_indices]
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, db_scores, marker='o', label='Davies-Bouldin Score')
    plt.scatter(top_k_values, top_db_scores, color='red', label='Top 5', zorder=5)
    plt.title('Davies-Bouldin Score for Different Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.grid(True)
    plt.legend()
    plt.show()









    


    
    
    

    
   
    

    







    

    










