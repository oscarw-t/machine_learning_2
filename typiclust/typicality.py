from sklearn.neighbors import NearestNeighbors


def compute_typicality(features, k=20):
        
    #typicality = 1 / mean dist to k neighbors
    #high score = dense region

    k_actual = min(k, len(features) - 1)

    nn_model = NearestNeighbors(n_neighbors=k_actual + 1, metric='euclidean')

    #train nearest neighbours
    nn_model.fit(features)

    distances, _ = nn_model.kneighbors(features)

    #distances[:,0] is self-distance, skip
    mean_dist = distances[:, 1:].mean(axis=1)
    return 1.0 / (mean_dist + 1e-10)