import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from preprocess import preprocess_adj
import numpy as np
from model import *
from tqdm import tqdm

class stMMC(torch.nn.Module):
    def __init__(self, 
            model_type,
            corrupted_graph,
            num_sample,
            adata,
            dropout,
            epochs,
            learning_rate,
            device,
            clustering_loss,
            rel_sigma,
            input_dim, 
            hidden_dim, 
            output_dim,
            num_layers,
            num_classes,
            loss_weights,
            weight_decay=0.0):
        super(stMMC, self).__init__()
        
        self.model_type = model_type
        self.corrupted_graph = corrupted_graph
        self.num_sample = num_sample
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.num_classes = num_classes
        self.rel_sigma = rel_sigma
        self.num_layers = num_layers
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.clustering_loss = clustering_loss

        # decide on which clustering loss to use
        if self.clustering_loss.lower() == 'ddc':
            self.clustering_loss_fn = DDC_loss(num_class=self.num_classes, num_samples=self.num_sample, device=self.device, rel_sigma=self.rel_sigma)
        elif self.clustering_loss.lower() == 'dec':
            self.clustering_loss_fn = DEC(num_clusters=self.num_classes, input_shape=self.output_dim)
        self.alpha = loss_weights[0]
        self.gamma = loss_weights[1]
        self.beta = loss_weights[2]
        
        self.adata = adata
        # Convert gene_data.X to dense tensor if it is sparse
        if isinstance(adata.X, sp.spmatrix):
            self.gene_feature = torch.tensor(adata.X.todense(), dtype=torch.float32).to(device)
        else:
            self.gene_feature = torch.tensor(adata.X, dtype=torch.float32).to(device)
        
        # Convert image_data to tensor if it's a DataFrame or NumPy array, and move to the device
        if isinstance(adata.obsm['image_feature'], pd.DataFrame):
            self.image_feature = torch.tensor(adata.obsm['image_feature'].values, dtype=torch.float32).to(device)
        elif isinstance(adata.obsm['image_feature'], np.ndarray):
            self.image_feature = torch.tensor(adata.obsm['image_feature'], dtype=torch.float32).to(device)
        else:
            self.image_feature = adata.obsm['image_feature'].to(device)
        self.spatial_graph_neigh = torch.FloatTensor(adata.obsm['spatial_graph_neigh'].copy() + np.eye(adata.obsm['spatial_adj'].shape[0])).to(self.device)
        self.gene_graph_neigh = torch.FloatTensor(adata.obsm['gene_graph_neigh'].copy() + np.eye(adata.obsm['gene_adj'].shape[0])).to(self.device)

        spatial_adj = self.adata.obsm['spatial_adj']
        spatial_adj = preprocess_adj(spatial_adj)
        self.spatial_adj = torch.FloatTensor(spatial_adj).to(self.device)
        gene_adj = self.adata.obsm['gene_adj']
        gene_adj = preprocess_adj(gene_adj)
        self.gene_adj = torch.FloatTensor(gene_adj).to(self.device)
        self.label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)

        if self.corrupted_graph == 0:
            if self.model_type == 1:
                self.encoder = parallel_encoder(input_dim, hidden_dim, output_dim, dropout, num_layers)
            elif self.model_type == 0:
                self.encoder = graph_encoder(input_dim, hidden_dim, output_dim, dropout, num_layers)
        else:
            if self.model_type == 1:
                self.model = parallel_encoder_with_contrast(input_dim, hidden_dim, output_dim, self.spatial_graph_neigh, self.gene_graph_neigh, self.num_layers, self.num_classes, self.clustering_loss, dropout = dropout).to(self.device)
            elif self.model_type == 0:
                self.model = Encoder(input_dim[0], output_dim, self.spatial_graph_neigh, dropout)

    def train_mm(self):
        self.loss_CSL = nn.BCEWithLogitsLoss()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        print('Training with both Histology Images and Spatial Transcriptomics data')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            
            if self.clustering_loss == 'none':
                self.hiden_feat, self.emb, ret1, ret2, ret1_a, ret2_a = self.model(self.gene_feature, self.image_feature, self.spatial_adj, self.gene_adj)
            else:
                self.hiden_feat, self.emb, ret1, ret2, ret1_a, ret2_a, cluster_assignments, x_hidden = self.model(self.gene_feature, self.image_feature, self.spatial_adj, self.gene_adj)
            
            self.loss_sl1_1 = self.loss_CSL(ret1, self.label_CSL)
            self.loss_sl1_2 = self.loss_CSL(ret1_a, self.label_CSL)

            self.loss_sl2_1 = self.loss_CSL(ret2, self.label_CSL)
            self.loss_sl2_2 = self.loss_CSL(ret2_a, self.label_CSL)

            self.loss_feat = F.mse_loss(self.gene_feature, self.emb)
            
            if self.clustering_loss.lower() == 'ddc':
                self.cluster_loss = self.clustering_loss_fn(cluster_assignments, x_hidden, self.num_classes)
            elif self.clustering_loss.lower() == 'dec':
                q = self.clustering_loss_fn(self.hiden_feat.to(self.device))
                p = DEC.target_distribution(q).detach()
                kld_loss = F.kl_div(q.log(), p, reduction='batchmean')
                self.cluster_loss = kld_loss
            else:
                self.cluster_loss = 0
            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl1_1 + self.loss_sl1_2 + self.loss_sl2_1 + self.loss_sl2_2) + self.gamma*self.cluster_loss
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Training Complete!")
        
        with torch.no_grad():
            self.model.eval()
            if self.clustering_loss == 'none':
                self.emb_rec = self.model(self.gene_feature, self.image_feature, self.spatial_adj, self.gene_adj)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
            else:
                _, self.emb_rec, _, _, _, _, self.cluster_assignments, _ = self.model(self.gene_feature, self.image_feature, self.spatial_adj, self.gene_adj)
                self.emb_rec = self.emb_rec.detach().cpu().numpy()
                self.cluster_result =  torch.argmax(self.cluster_assignments, dim=1).detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
                self.adata.obs['cluster'] = self.cluster_result
            return self.adata

    def train_sm(self):
        self.loss_CSL = nn.BCEWithLogitsLoss()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        print('Training with Only Spatial Transcriptomics data...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.gene_feature, self.spatial_adj)
            
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.gene_feature, self.emb)
            
            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)
            print(f'Epoch {epoch}: Loss {loss}')
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Training Complete!")
        
        with torch.no_grad():
            self.model.eval()
            self.emb_rec = self.model(self.gene_feature, self.spatial_adj)[1].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec
                
            return self.adata