import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score

def graficar_clusters (X_cluster, y_cluster,titulo,x_label,y_label):
    colores = ['blue','red','green','purple','brown','black','yellow','orange']
    labels = np.unique(y_cluster)
    for i in range(0,len(labels)):
        plt.scatter(X_cluster[y_cluster==labels[i],0],X_cluster[y_cluster==labels[i],1],s=100, c=colores[i])
    plt.title(titulo)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(labels)
    plt.show()
    
    unique, counts = np.unique(y_cluster, return_counts=True)
    print(unique)
    print(counts)

def graficar_clusters_3d (X_cluster, y_cluster,titulo,x_label,y_label,horizontal,vertical):
    labels = list(np.unique(y_cluster))
    colores = ['blue','red','green','purple','brown','black','yellow','orange']
    asignar=[]
    for i in range(0,len(y_cluster)):
        color = labels.index(y_cluster[i])
        asignar.append(colores[color])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_cluster[:, 0], X_cluster[:, 1], X_cluster[:, 2], c=asignar,s=60)  
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.view_init(vertical,horizontal )
    
    
def ajustar_kmeans(X_clusters,valor_x_linea_vertical,label_x,label_y,rotacion_horizontal,rotacion_vertical):
    wcss = []
    vmeasure_score =[]
    km_silhouette = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter = 500, n_init = 30,random_state=0,n_jobs=2)
        kmeans.fit(X_clusters)
        preds = kmeans.predict(X_clusters)
        
        if i>1:
            #Silhoutte method to determine number of clusters
            silhouette = silhouette_score(X_clusters,preds)
            km_silhouette.append(silhouette)
                
        #within cluster sum of squares
        wcss.append(kmeans.inertia_)
        
    plt.plot(range(1,11),wcss)
    plt.title('Elbow Method for determining \nnumber of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.axvline(x=valor_x_linea_vertical,c='red')
    plt.xticks((range(1,11)))
    plt.show()
    
    plt.figure(figsize=(7,4))
    plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
    plt.scatter(x=[i for i in range(2,11)],y=km_silhouette,s=150,edgecolor='k')
    plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("Silhouette score",fontsize=15)
    plt.xticks([i for i in range(2,11)],fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()
    
    kmeans = KMeans(n_clusters=valor_x_linea_vertical, init='k-means++', max_iter = 500, n_init = 30,random_state=0,n_jobs=2)
    y_kmeans=kmeans.fit_predict(X_clusters)
    graficar_clusters(X_cluster=X_clusters,y_cluster=y_kmeans,titulo="clusters k-means",x_label=label_x,y_label=label_y)
    graficar_clusters_3d(X_cluster=X_clusters,y_cluster=y_kmeans,titulo="clusters_k-means",x_label=label_x,y_label=label_y,horizontal=rotacion_horizontal,vertical=rotacion_vertical)
         
    return y_kmeans

def ajustar_shift_means(X_clusters,label_x,label_y,rotacion_horizontal,rotacion_vertical):
    #Cluster por mean shift
    bandwidth = estimate_bandwidth(X_clusters, quantile=0.2, n_samples=500,random_state=0)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X_clusters)
    y_ms = ms.labels_
    cluster_centers_ms = ms.cluster_centers_
    labels_unique_ms = np.unique(y_ms)
    n_clusters_ms = len(labels_unique_ms)

    print("number of estimated clusters : %d" % n_clusters_ms)
    
    graficar_clusters(X_cluster=X_clusters,y_cluster=y_ms,titulo="clusters k-means",x_label=label_x,y_label=label_y)
    graficar_clusters_3d(X_cluster=X_clusters,y_cluster=y_ms,titulo="clusters_k-means",x_label=label_x,y_label=label_y,horizontal=rotacion_horizontal,vertical=rotacion_vertical)
    return(y_ms)
    
def ajustar_gaussian_mixture(X_clusters,number_of_components,random_st,label_x,label_y,rotacion_horizontal,rotacion_vertical):
    # training gaussian mixture model 
    
    gm_bic= []
    for i in range(2,12):
        gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(X_clusters)
        gm_bic.append(gm.bic(X_clusters))
    
    plt.figure(figsize=(7,4))
    plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
    plt.scatter(x=[i for i in range(2,12)],y=np.log(gm_bic),s=150,edgecolor='k')
    plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
    plt.xticks([i for i in range(2,12)],fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()  
    
    gmm = GaussianMixture(n_components=number_of_components,random_state = random_st)
    gmm.fit(X_clusters)
    #predictions from gmm
    y_gauss = gmm.predict(X_clusters)
    graficar_clusters(X_cluster=X_clusters,y_cluster=y_gauss,titulo="clusters k-means",x_label=label_x,y_label=label_y)
    graficar_clusters_3d(X_cluster=X_clusters,y_cluster=y_gauss,titulo="clusters_k-means",x_label=label_x,y_label=label_y,horizontal=rotacion_horizontal,vertical=rotacion_vertical)
    return y_gauss

def ajustar_cluster_jerarquico(X_clusters,metodo,x_label,y_label,num_clusters,rotacion_horizontal,rotacion_vertical):
    dendogram=sch.dendrogram(sch.linkage(X_clusters,method=metodo))
    plt.title('Dendogram')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(y=num_clusters,c='red')
    plt.show()
        
    hc = AgglomerativeClustering(n_clusters=num_clusters, affinity='l2', linkage='complete')
    y_hc = hc.fit_predict(X_clusters)
    graficar_clusters(X_cluster=X_clusters,y_cluster=y_hc,titulo="clusters k-means",x_label=x_label,y_label=y_label)
    graficar_clusters_3d(X_cluster=X_clusters,y_cluster=y_hc,titulo="clusters_k-means",x_label=x_label,y_label=y_label,horizontal=rotacion_horizontal,vertical=rotacion_vertical)
    return y_hc

def graficar_componentes_principales (features, base, respuesta):
    #Estandarizar predictores
    X_PCA = base.loc[:, features].values
    X_PCA = StandardScaler().fit_transform(X_PCA)
    #Ajustar PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_PCA)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame({"target":respuesta})], axis = 1)
    #Graficar variable de respuesta en el plano de los componentes
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0, 1]
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    print(pca.explained_variance_ratio_)
