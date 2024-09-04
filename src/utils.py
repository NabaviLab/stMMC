import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import ot
import scanpy as sc

def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    
    pca = PCA(n_components=20, random_state=42) 
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding
    
    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
    
    print('Clustering without refinement is done!')
    # calculate metric ARI
    ARI = adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    NMI = normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
    adata.uns['ARI'] = ARI
    adata.uns['NMI'] = NMI
    print('ARI:', ARI)
    print('NMI:', NMI)
    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['smoothed_domain'] = new_type
        ARI = adjusted_rand_score(adata.obs['smoothed_domain'], adata.obs['ground_truth'])
        NMI = normalized_mutual_info_score(adata.obs['smoothed_domain'], adata.obs['ground_truth'])
        print('ARI after refinement:', ARI)
        print('NMI after refinement:', NMI)
       
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    


# Do a evaluation of the model clustering results using ARI and NMI
def clustering_metrics(cluster_assignments, true_labels):
    # Check if the input: cluster_assignments are the correct format
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.detach().cpu().numpy()
    elif isinstance(cluster_assignments, pd.Series):
        cluster_assignments = cluster_assignments.values
    elif isinstance(cluster_assignments, np.ndarray):
        cluster_assignments = cluster_assignments
    else:
        raise ValueError("The cluster_assignments should be a torch.Tensor, pd.Series, or np.ndarray.")

    ari = adjusted_rand_score(true_labels, cluster_assignments)
    nmi = normalized_mutual_info_score(true_labels, cluster_assignments)
    
    print(f"ARI: {ari}")
    print(f"NMI: {nmi}")
    
    return ari, nmi

# smoothing clustering result using spatial graph
def smoothing_cluster(cluster_assignments, spatial_graph):
    # Check if the input: cluster_assignments are the correct format
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.detach().cpu().numpy()
    elif isinstance(cluster_assignments, pd.Series):
        cluster_assignments = cluster_assignments.values
    elif isinstance(cluster_assignments, np.ndarray):
        cluster_assignments = cluster_assignments
    else:
        raise ValueError("The cluster_assignments should be a torch.Tensor, pd.Series, or np.ndarray.")
    
    # Check if the input: spatial_graph are the correct format
    if isinstance(spatial_graph, torch.Tensor):
        spatial_graph = spatial_graph.detach().cpu().numpy()
    elif isinstance(spatial_graph, pd.DataFrame):
        spatial_graph = spatial_graph.values
    elif isinstance(spatial_graph, np.ndarray):
        spatial_graph = spatial_graph
    else:
        raise ValueError("The spatial_graph should be a torch.Tensor, pd.DataFrame, or np.ndarray.")
    
    # Get the number of nodes
    num_nodes = cluster_assignments.shape[0]
    
    # Initialize the new cluster assignments
    new_cluster_assignments = np.zeros(num_nodes)
    
    for i in range(num_nodes):
        # Get the neighbors of node i
        neighbors = np.where(spatial_graph[i] == 1)[0]
        
        # Get the cluster assignments of the neighbors
        neighbor_cluster_assignments = cluster_assignments[neighbors]
        
        # Get the unique cluster assignments
        unique_clusters, counts = np.unique(neighbor_cluster_assignments, return_counts=True)
        
        # Get the cluster assignment with the maximum count
        max_cluster = unique_clusters[np.argmax(counts)]
        
        # Assign the node i to the cluster with the maximum count
        new_cluster_assignments[i] = max_cluster
        
    return new_cluster_assignments