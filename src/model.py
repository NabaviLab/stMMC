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

class parallel_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_layers):
        super(parallel_encoder, self).__init__()
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_dim[0], hidden_dim)
        self.fc2 = nn.Linear(input_dim[1], hidden_dim)
        if self.num_layers == 1:
            self.gcn1 = GCNConv(hidden_dim, output_dim)
            self.gcn2 = GCNConv(hidden_dim, output_dim)
        elif self.num_layers == 2:
            self.gcn1_1 = GCNConv(hidden_dim, hidden_dim // 2)
            self.gcn1_2 = GCNConv(hidden_dim, hidden_dim // 2)
            self.gcn2_1 = GCNConv(hidden_dim // 2, output_dim)
            self.gcn2_2 = GCNConv(hidden_dim // 2, output_dim)

        # Dropout rate
        self.dropout = dropout

        # Define learnable weights
        self.w1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))  # Initialize with 0.5
        self.w2 = nn.Parameter(torch.tensor(0.5, requires_grad=True))  # Initialize with 0.5

    def forward(self, gene_data, image_data, gene_edge_index, spatial_edge_index):
        x1 = torch.relu(self.fc1(gene_data))
        x2 = torch.relu(self.fc2(image_data))
        
        # Apply dropout to the features before passing them to the GCN layers
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        if self.num_layers == 1:
            x1 = self.gcn1(x1, spatial_edge_index)
            x2 = self.gcn2(x2, gene_edge_index)
            x = self.w1 * x1 + (1-self.w1) * x2
        elif self.num_layers == 2:
            x1 = torch.relu(self.gcn1_1(x1, spatial_edge_index))
            x2 = torch.relu(self.gcn1_2(x2, gene_edge_index))

            x = self.w1 * x1 + (1-self.w1) * x2
            
            x1 = self.gcn2_1(x, spatial_edge_index)
            x2 = self.gcn2_2(x, gene_edge_index)
            
            x = self.w2 * x1 + (1-self.w2) * x2
        return x

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 

class graph_encoder_with_contrastive(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_layers, spatial_graph_neigh):
        super(graph_encoder_with_contrastive, self).__init__()
        self.num_layers = num_layers
        self.spatial_graph_neigh = spatial_graph_neigh
        # self.fc1 = nn.Linear(input_dim[0], hidden_dim)
        if num_layers == 2:
            self.gcn1 = GCNConv(input_dim[0], hidden_dim // 2)
            self.gcn2 = GCNConv(hidden_dim // 2, output_dim)
        elif num_layers == 1:
            self.gcn1 = GCNConv(input_dim[0], output_dim)
        else:
            raise Exception("Invalid number of layers. Must be 1 or 2.")
        self.disc1 = Discriminator(output_dim)
        self.read = AvgReadout()

        # Dropout rate
        self.dropout = dropout

    def permutation(self, data):
        # Permutation function to shuffle node features
        ids = torch.randperm(data.size(0))  # Generate a random permutation of indices
        permuted_data = data[ids]  # Permute the rows of the data
        return permuted_data

    def forward(self, gene_data, spatial_edge_index):
        x1 = gene_data
        x1_c = self.permutation(x1)
        
        # Apply dropout to the features before passing them to the GCN layers
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1_c = F.dropout(x1_c, p=self.dropout, training=self.training)
        
        # x1 = F.relu(self.fc1(gene_data))
        if self.num_layers == 2:
            x1 = torch.relu(self.gcn1(x1, spatial_edge_index))
            x1_c = torch.relu(self.gcn1(x1_c, spatial_edge_index))
            
            x1 = torch.relu(self.gcn2(x1, spatial_edge_index))
            x1_c = torch.relu(self.gcn2(x1_c, spatial_edge_index))
        elif self.num_layers == 1:
            x1 = torch.relu(self.gcn1(x1, spatial_edge_index))
            x1_c = torch.relu(self.gcn1(x1_c, spatial_edge_index))
        else:
            raise Exception("Invalid number of layers. Must be 1 or 2.")

        global_average_1 = F.sigmoid(self.read(emb=x1, mask=self.spatial_graph_neigh))
        global_average_1_c = F.sigmoid(self.read(emb=x1_c, mask=self.spatial_graph_neigh))

        ret1 = self.disc1(global_average_1, x1, x1_c)
        ret1_c = self.disc1(global_average_1_c, x1_c, x1)
        return x1, ret1, ret1_c

class graph_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_layers):
        super(graph_encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        if self.num_layers == 1:
            self.conv1 = GCNConv(input_dim[0], output_dim)
        elif self.num_layers == 2:
            self.conv1 = GCNConv(input_dim[0], hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            x = self.conv1(x, edge_index)
        elif self.num_layers == 2:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
        return x

class decoder(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, way = 'gcn'):
        super(decoder, self).__init__()
        self.way = way
        if way == 'fcnn':
            self.batch_size = batch_size
            self.fc1 = nn.Linear(batch_size * input_dim, batch_size * hidden_dim)
            self.fc2 = nn.Linear(batch_size * hidden_dim, batch_size * output_dim)
        elif way == 'gcn':
            self.gcn = GCNConv(input_dim, output_dim)
        else:
            raise Exception("Invalid way. Must be 'fcnn' or 'gcn'.")

    def forward(self, x, spatial_edge_index=None):
        if self.way == 'fcnn':
            x = x.view(-1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = x.view(self.batch_size, -1)
        elif self.way == 'gcn':
            x = F.relu(self.gcn(x, spatial_edge_index))
        return x
    
    def forward_with_batches(self, x, batch_size=64):
        """
        Forward pass with mini-batching support.
        
        Args:
            x (Tensor): Input tensor with shape [num_samples, input_dim]
            batch_size (int): The size of each mini-batch
        
        Returns:
            Tensor: The output after processing in mini-batches.
        """
        num_samples = x.size(0)
        outputs = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            # Extract the batch
            batch = x[start_idx:end_idx]

            # Pad the batch if it's smaller than the batch_size
            if batch.size(0) < batch_size:
                padding_size = batch_size - batch.size(0)
                # Create a padding tensor with the same number of dimensions as the batch and filled with zeros
                padding = torch.zeros((padding_size, *batch.size()[1:]), device=batch.device)
                # Concatenate the batch with the padding
                batch = torch.cat([batch, padding], dim=0)

            # Forward pass for the batch
            batch_output = self.forward(batch)

            # Remove the padding from the output if the batch was padded
            if batch.size(0) > end_idx - start_idx:
                batch_output = batch_output[:(end_idx - start_idx)]
            
            outputs.append(batch_output)

        # Concatenate all batch outputs
        return torch.cat(outputs, dim=0)

class DDC_loss(nn.Module):
    def __init__(self, num_class, num_samples, device, rel_sigma=0.15):
        super(DDC_loss, self).__init__()
        self.EPSILON = 1E-9
        self.num_class = num_class
        self.num_samples = num_samples
        self.device = device
        self.eye = torch.eye(num_class, device=self.device)
        self.rel_sigma = rel_sigma

    def triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _atleast_epsilon(self, X, eps=1E-9):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def cdist(self, X, Y):
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X**2, dim=1, keepdim=True)
        y2 = torch.sum(Y**2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    def d_cs(self, A, K, num_class):
        """
        Cauchy-Schwarz divergence.
        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)
        nom = self._atleast_epsilon(nom)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=self.EPSILON**2)
        d = 2 / (num_class * (num_class - 1)) * self.triu(nom / torch.sqrt(dnom_squared))
        return d

    def kernel_from_distance_matrix(self, dist, rel_sigma, min_sigma=1E-9):
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = nn.functional.relu(dist)
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k

    def vector_kernel(self, x, rel_sigma=0.15):
        return self.kernel_from_distance_matrix(self.cdist(x, x), rel_sigma)

    def forward(self, cluster_assignment, x_hidden, num_class):
        hidden_kernel = self.vector_kernel(x_hidden, self.rel_sigma)
        m = torch.exp(-self.cdist(cluster_assignment, self.eye))

        ddc1 = self.d_cs(cluster_assignment, hidden_kernel, self.num_class)
        ddc2 = 2 / (self.num_samples * (self.num_samples - 1)) * self.triu(cluster_assignment @ torch.t(cluster_assignment))
        ddc3 = self.d_cs(m, hidden_kernel, num_class)
        # print(ddc1, ddc2, ddc3)
        return ddc1+ddc2+ddc3

class DEC(nn.Module):
    def __init__(
        self, 
        num_clusters:int,
        input_shape:int,
        alpha: float = 1.0,
        cluster_centers = None
        ):
        super(DEC, self).__init__()

        self.num_clusters = num_clusters
        self.alpha = alpha
        self.input_shape = input_shape

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.num_clusters, self.input_shape, dtype=torch.float32)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.clustcenters = nn.Parameter(initial_cluster_centers)

    def forward(self, inputs):
        """ student t-distribution, as same as used in t-SNE algorithm.
            inputs: the variable containing data, shape=(n_samples, n_features)
            output: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, axis=1) - self.clustcenters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, axis=1), 0, 1)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

class clustering_network(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super(clustering_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, num_clusters)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x_hidden = x
        x = F.softmax(self.fc2(x), dim=1)
        return x, x_hidden  # Output soft cluster assignments
    
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        return logits
    
class Encoder(nn.Module):
    def __init__(self, in_features, 
                    out_features, 
                    graph_neigh, 
                    dropout=0.0, 
                    act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
    
    def permutation(self, data):
        # Permutation function to shuffle node features
        ids = torch.randperm(data.size(0))  # Generate a random permutation of indices
        permuted_data = data[ids]  # Permute the rows of the data
        return permuted_data
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, adj):
        feat_a = self.permutation(feat)
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)
        
        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a

class parallel_encoder_with_contrast(nn.Module):
    def __init__(self, in_features, 
                    hidden_dim, 
                    output_dim, 
                    graph_neigh1, 
                    graph_neigh2, 
                    num_layers, 
                    num_classes,
                    clustering_loss,
                    dropout=0.0, act=F.relu):
        super(parallel_encoder_with_contrast, self).__init__()
        self.in_features1 = in_features[0]
        self.in_features2 = in_features[1]
        self.num_layers = num_layers
        # self.fc1 = nn.Linear(self.in_features1, hidden_dim)
        # self.fc2 = nn.Linear(self.in_features2, hidden_dim)
        self.output_dim = output_dim
        self.graph_neigh1 = graph_neigh1
        self.graph_neigh2 = graph_neigh2
        self.num_classes = num_classes
        self.dropout = dropout
        self.clustering_loss = clustering_loss
        self.act = act
        if clustering_loss != 'none':
            self.cluster = clustering_network(self.output_dim, self.num_classes)
        
        if num_layers == 1:
            self.weight1_1 = nn.Parameter(torch.FloatTensor(self.in_features1, self.output_dim))
            self.weight1_2 = nn.Parameter(torch.FloatTensor(self.in_features2, self.output_dim))
            self.weight2 = nn.Parameter(torch.FloatTensor(self.output_dim, self.in_features1))
            self.reset_parameters()

            self.w1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))  # Initialize with 0.5
        elif num_layers == 2:
            self.weight1_1 = nn.Parameter(torch.FloatTensor(self.in_features1, hidden_dim // 2))
            self.weight1_2 = nn.Parameter(torch.FloatTensor(self.in_features2, hidden_dim // 2))
            self.weight2_1 = nn.Parameter(torch.FloatTensor(hidden_dim // 2, self.output_dim))
            self.weight2_2 = nn.Parameter(torch.FloatTensor(hidden_dim // 2, self.output_dim))
            self.weight3 = nn.Parameter(torch.FloatTensor(self.output_dim, self.in_features1))
            self.reset_parameters()

            self.w1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))  # Initialize with 0.5
            self.w2 = nn.Parameter(torch.tensor(0.5, requires_grad=True))  # Initialize with 0.5
        
        self.disc1 = Discriminator(self.output_dim)
        self.disc2 = Discriminator(self.output_dim)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
    
    def permutation(self, data):
        # Permutation function to shuffle node features
        ids = torch.randperm(data.size(0))  # Generate a random permutation of indices
        permuted_data = data[ids]  # Permute the rows of the data
        return permuted_data
    
    def reset_parameters(self):
        if self.num_layers == 1:
            torch.nn.init.xavier_uniform_(self.weight1_1)
            torch.nn.init.xavier_uniform_(self.weight1_2)
            torch.nn.init.xavier_uniform_(self.weight2)
        elif self.num_layers == 2:
            torch.nn.init.xavier_uniform_(self.weight1_1)
            torch.nn.init.xavier_uniform_(self.weight1_2)
            torch.nn.init.xavier_uniform_(self.weight2_1)
            torch.nn.init.xavier_uniform_(self.weight2_2)
            torch.nn.init.xavier_uniform_(self.weight3)

    def forward(self, feat1, feat2, adj1, adj2):
        # feat1 = self.fc1(feat1)
        # feat2 = self.fc2(feat2)
        feat1_a = self.permutation(feat1)
        feat2_a = self.permutation(feat2)
        z1 = F.dropout(feat1, self.dropout, self.training)
        z1 = torch.mm(z1, self.weight1_1)
        z1 = torch.mm(adj1, z1)
        
        z2 = F.dropout(feat2, self.dropout, self.training)
        z2 = torch.mm(z2, self.weight1_2)
        z2 = torch.mm(adj2, z2)

        hiden_emb1 = z1
        hiden_emb2 = z2
        emb1 = self.act(z1)
        emb2 = self.act(z2)

        hiden_emb = self.w1 * hiden_emb1 + (1-self.w1) * hiden_emb2
        
        if self.num_layers == 1:
            h = torch.mm(hiden_emb, self.weight2)
            h = torch.mm(adj1, h)
        elif self.num_layers == 2:
            z1 = torch.mm(hiden_emb, self.weight2_1)
            z1 = torch.mm(adj1, z1)

            z2 = torch.mm(hiden_emb, self.weight2_2)
            z2 = torch.mm(adj2, z2)

            hiden_emb1 = z1
            hiden_emb2 = z2
            emb1 = self.act(z1)
            emb2 = self.act(z2)

            hiden_emb = self.w2 * hiden_emb1 + (1-self.w2) * hiden_emb2

            h = torch.mm(hiden_emb, self.weight3)
            h = torch.mm(adj1, h)
        
        z1_a = F.dropout(feat1_a, self.dropout, self.training)
        z1_a = torch.mm(z1_a, self.weight1_1)
        z1_a = torch.mm(adj1, z1_a)
        emb1_a = self.act(z1_a)

        z2_a = F.dropout(feat2_a, self.dropout, self.training)
        z2_a = torch.mm(z2_a, self.weight1_2)
        z2_a = torch.mm(adj2, z2_a)
        emb2_a = self.act(z2_a)
        
        if self.num_layers ==2:
            hiden_emb_a = self.w1 * z1_a + (1-self.w1) * z2_a
            z1_a = torch.mm(hiden_emb_a, self.weight2_1)
            z1_a = torch.mm(adj1, z1_a)
            emb1_a = self.act(z1_a)

            z2_a = torch.mm(hiden_emb_a, self.weight2_2)
            z2_a = torch.mm(adj2, z2_a)
            emb2_a = self.act(z2_a)

            hiden_emb_a = self.w2 * z1_a + (1-self.w2) * z2_a
        g1 = self.read(emb1, self.graph_neigh1) 
        g1 = self.sigm(g1)  

        g1_a = self.read(emb1_a, self.graph_neigh1)
        g1_a = self.sigm(g1_a)  

        g2 = self.read(emb2, self.graph_neigh2)
        g2 = self.sigm(g2)

        g2_a = self.read(emb2_a, self.graph_neigh2)
        g2_a = self.sigm(g2_a)

        ret1 = self.disc1(g1, emb1, emb1_a)  
        ret1_a = self.disc1(g1_a, emb1_a, emb1) 

        ret2 = self.disc2(g2, emb2, emb2_a)
        ret2_a = self.disc2(g2_a, emb2_a, emb2)

        if self.clustering_loss != 'none':
            cluster_assignments, x_hidden = self.cluster(hiden_emb)
            return hiden_emb, h, ret1, ret1_a, ret2, ret2_a, cluster_assignments, x_hidden
        else:
            return hiden_emb, h, ret1, ret1_a, ret2, ret2_a