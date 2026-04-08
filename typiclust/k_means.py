from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster(features, n_clusters):
    #kmeans for small k, minibatch for large k
    

    if n_clusters <= 50:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) #CHANGE BACK TO 42 FOR SUBMISSION

    else:
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, #HERE TOO
                                batch_size=1024)
    
    return model.fit_predict(features)
