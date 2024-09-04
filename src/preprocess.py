import os
import ot
import torch
import random
import ast
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from sklearn.decomposition import PCA
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from sklearn.preprocessing import LabelEncoder

def preprocess(adata, n_gene=3000):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_gene)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def load_data(data_dir, genome_file_path, spatial_file_path, image_file_path, true_label_path, n_gene, n_neighbors):
    # adata = sc.read_10x_h5(genome_file_path)
    ground_truth_nan = False
    adata = sc.read_visium(data_dir, count_file=genome_file_path, load_images=True)
    adata.var_names_make_unique()

    adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)
    preprocess(adata, n_gene=n_gene)

    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]

    if image_file_path is not None:
        # read image feature data
        image_data = pd.read_csv(image_file_path, header=0, index_col=0)
        adata = adata[adata.obs_names.isin(image_data.index.tolist())]
        image_data = image_data.loc[adata.obs_names]

        image_data['feature_vector'] = image_data['feature_vector'].apply(ast.literal_eval)
        image_data_new = pd.DataFrame(image_data['feature_vector'].tolist(), index=image_data.index)
    else:
        image_data_new = None

    adata = adata.copy()

    if 'spatial' not in adata.obsm.keys():
        spatial_location = pd.read_csv(spatial_file_path, sep=',', header=0)
        tmp = pd.DataFrame({'barcode':adata.obs_names.to_list()})

        spatial_location_in_tissue_sorted = pd.merge(tmp, spatial_location[spatial_location['in_tissue'] == 1], left_on='barcode', right_on='barcode')
        adata.obsm['spatial'] = spatial_location_in_tissue_sorted[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
        
        construct_spatial_interaction(adata, n_neighbors=n_neighbors)
    else:
        construct_spatial_interaction(adata, n_neighbors=n_neighbors)
    
    construct_genome_interaction(adata, n_neighbors=n_neighbors)

    if 'label_CSL' not in adata.obsm.keys():    
        add_contrastive_label(adata)

    # Load the ground truth label of the dataset
    if true_label_path.endswith('.tsv'):
        true_labels = pd.read_csv(true_label_path, sep='\t', header=0, index_col=0)
    elif true_label_path.endswith('.csv'):
        true_labels = pd.read_csv(true_label_path, header=0, index_col=0)
    elif true_label_path.endswith('.txt'):
        true_labels = pd.read_csv(true_label_path, sep='\t', header=0, index_col=0)
    true_labels = true_labels.loc[adata.obs_names]
    # check if there is nan in the ground truth label
    if true_labels['Cluster'].isnull().sum() > 0:
        ground_truth_nan = True
    le = LabelEncoder()
    le.fit(true_labels['Cluster'])
    true_labels['Cluster'] = le.transform(true_labels['Cluster'])
    # convert the ground truth label to integer
    adata.obs['ground_truth'] = true_labels['Cluster'].values

    # load image feature into anndata
    if 'image_feature' not in adata.obsm.keys():
        adata.obsm['image_feature'] = image_data_new.values
    return adata, image_data_new, ground_truth_nan

def construct_spatial_interaction(adata, n_neighbors=3):
    ## Constructing spot-to-spot location-based interactive graph
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['spatial_distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['spatial_graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    np.fill_diagonal(adj, 1)
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['spatial_adj'] = adj

def construct_genome_interaction(adata, n_neighbors=3):
    # Constructing spot-to-spot genome-based interactive graph
    if sp.issparse(adata.X):
        gene_data = adata.X.toarray()
    else:
        gene_data = adata.X
    
    # reduce dimension by PCA
    n_components = 50  # Number of components to keep after PCA
    pca = PCA(n_components=n_components)
    gene_data_pca = pca.fit_transform(gene_data)

    # calculate distance matrix
    distance_matrix = ot.dist(gene_data_pca, gene_data_pca, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['gene_distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['gene_graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    np.fill_diagonal(adj, 1)
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['gene_adj'] = adj

def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 