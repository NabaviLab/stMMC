import torch
import argparse
import json
import ast
import torch.nn as nn
import torch.nn.functional as F
from model import *
from preprocess import *
import datetime
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from model import DEC
from stMMC import stMMC
from utils import clustering

# load the args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_feature_length', type=int, default=512, help='Length of the hidden feature vector')
    parser.add_argument('--num_gene', type=int, default=3000, help='Number of genes to keep')
    parser.add_argument('--num_neighbors', type=int, default=3, help='Number of neighbors for spatial graph')
    parser.add_argument('--num_spatial_class', type=int, default=0, help='Number of spatial classes in the dataset, 0 for using the number of classes in true labels')
    parser.add_argument('--device', type=int, default=0, help='Which GPU to use.')
    parser.add_argument('--epoch', type=int, default=600, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model')
    parser.add_argument('--clustering_loss', type=str, default='none', help='Loss function for clustering: ddc, dec or none')
    parser.add_argument('--rel_sigma', type=float, default=0.15, help='Sigma for the relative distance in DDC loss')
    parser.add_argument('--dataset', type=str, default='dlpfc_151507', help='Dataset identifier in the config file')
    parser.add_argument('--model', type=int, default=1, help='2 for mannual graph encoder, 1 for including image data, 0 for not including image data')
    parser.add_argument('--corrupted_graph', type=int, default=1, help='1 for using corrupted graph, 0 for not using corrupted graph')
    parser.add_argument('--loss_weights', type=str, default='[10, 0, 1]', help='Weights for the loss function [rec, clust, contrast]')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the 1 modality with corrputed graph model')
    parser.add_argument('--seed', type=int, default=39, help='Random seed for reproducibility')
    parser.add_argument('--radius', type=int, default=50, help='Radius for cluster smoothing')
    parser.add_argument('--conventional_method', type=str, default='mclust', help='mclust or gmm for using conventional clustering, none for not using conventional clustering')
    parser.add_argument('--save_model', type=int, default=0, help='0 is to not save the model and 1 is to save the model')
    parser.add_argument('--save_result', type=int, default=0, help='0 is to not save the result and 1 is to save the result')
    args = parser.parse_args()
    return args

def set_seed(seed):
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = get_args()

    set_seed(args.seed)
    # Set the device to GPU if available, otherwise to CPU
    if torch.cuda.is_available():
        device = torch.device('cuda', int(args.device))
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # load the dataset configuration file
    with open('src/dataset_config.json') as f:
        dataset_config = json.load(f)[args.dataset]

    # load the dataset
    data_folder_path = dataset_config['data_dir']
    gene_data_path = dataset_config['genome_path']
    spatial_data_path = os.path.join(dataset_config['data_dir'],dataset_config['spatial_path'])
    image_file_path = os.path.join(dataset_config['data_dir'],dataset_config['image_file_path'])
    true_label_path = os.path.join(dataset_config['data_dir'], dataset_config['ground_truth_path'])

    gene_data, image_data, ground_truth_nan = load_data(data_folder_path, gene_data_path, spatial_data_path, image_file_path, true_label_path, args.num_gene, args.num_neighbors)

    # check the number of the unique class in the true label and if it matches with the number of spatial classes
    if args.num_spatial_class == 0:
        # check if there is any class in the ground truth label class smaller than 20
        if ground_truth_nan:
            print('There are nan values in the ground truth label. Proceeding with n-1 classes.')
            args.num_spatial_class = len(gene_data.obs['ground_truth'].unique())-1
        else:
            args.num_spatial_class = len(gene_data.obs['ground_truth'].unique())

    elif len(gene_data.obs['ground_truth'].unique()) != args.num_spatial_class:
        print(f'Number of unique classes in the true label is {len(gene_data.obs["ground_truth"].unique())} but the number of spatial classes is {args.num_spatial_class}')
        print('Procceeding with caution')
    
    # Print shape of gene data and image data
    print("Gene data shape: ", gene_data.X.shape)
    print("Image data shape: ", image_data.shape)

    # Instantiate the model
    model = stMMC(model_type=args.model,
                    corrupted_graph=args.corrupted_graph,
                    num_sample=gene_data.X.shape[0],
                    adata = gene_data,
                    dropout=args.dropout,
                    epochs=args.epoch,
                    learning_rate=args.lr,
                    device=device,
                    clustering_loss=args.clustering_loss,
                    rel_sigma=args.rel_sigma,
                    input_dim=(gene_data.X.shape[1], image_data.shape[1]),
                    hidden_dim=256,
                    output_dim=32,
                    num_layers=args.num_layers,
                    num_classes=args.num_spatial_class,
                    loss_weights=ast.literal_eval(args.loss_weights)).to(device)

    if args.model == 1:
        result_data = model.train_mm()
    elif args.model == 0:
        result_data = model.train_sm()
    
    clustering(gene_data, n_clusters=args.num_spatial_class, radius=args.radius, key='emb', method=args.conventional_method, refinement=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save_model == 1:
        final_save_path = os.path.join(dataset_config['data_dir'], f'final_model_{timestamp}.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f'Final model saved to {final_save_path}')

    # save the cluster assignments and results
    if args.save_result == 1:
        result_data.write(os.path.join(dataset_config['result_dir'], f'result_data_{timestamp}.h5ad'))

if __name__ == '__main__':
    main() # Call the main function