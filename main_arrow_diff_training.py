import os
import random
import logging
import numpy as np
import torch

from torch_geometric.data import Data
# from torch.utils.tensorboard import SummaryWriter

from arrow_diff.data import load_data, train_val_test_split_graph
from arrow_diff.utils import read_config_file, initialize_logging, save_config_to_file
from arrow_diff.positional_encoding import positional_encoding
from arrow_diff.dm_trainer import DiffusionModelTrainer
from arrow_diff.unet import UNetAdapter

from arrow_diff.gnn_trainer import GNNTrainer
from arrow_diff.gcn import GCN


def main() -> None:
    """
    Main function.
    """
    os.chdir('../arrow-diff/')

    # 1) Set Up The Experiment

    # Read config file
    config = read_config_file('configs/config.yaml')

    dataset_name = config['data']['dataset']

    save_path = f'./results/dg_rw_diffusion' + ('_wo_features' if config['node_feature_dim'] else '') + f'/{dataset_name.lower()}'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Initialize logging
    initialize_logging(save_path, experiment_name='logging')

    logging.info(f'Config:\n{config}')

    # Save the config to a file
    save_config_to_file(config, save_path)

    seed = config['seed']

    if seed is not None:
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    # 2) Load The Data

    # Load the single graph or its largest connected component from the dataset
    data = load_data(dataset_name, path=config['data']['path'], seed=seed)

    logging.info(f'\nGraph (LCC):\n{data}')

    num_nodes = data.num_nodes

    if config['node_feature_dim'] or dataset_name == 'SBM':
        # Use positional encodings as node features
        data.x = positional_encoding(torch.arange(num_nodes), config['node_feature_dim'])

    # Split the edges of the graph into training edges, validation edges and non-edges, and testing edges and
    # non-edges
    train_edge_index, val_edge_index, val_non_edge_index, test_edge_index, test_non_edge_index = \
        train_val_test_split_graph(data, seed=seed)

    # Create the Data object that represents the training graph
    data_train = Data(edge_index=train_edge_index, num_nodes=num_nodes)

    # 3) Diffusion Model Training

    # Initialize the SummaryWriter
    writer = None  # SummaryWriter(log_dir=save_path, flush_secs=10)

    # Initialize the trainer
    dm_trainer = DiffusionModelTrainer(config['dm_training'], writer=writer)

    if seed is not None:
        # Use a seed for reproducible initialization of UNetAdapter
        torch.manual_seed(seed)

    # Initialize the model
    diffusion_model = UNetAdapter(config['diffusion_model']['hidden_channels'], num_nodes,
                                  config['diffusion_model']['node_embedding_dim'],
                                  config['dm_training']['num_diffusion_steps'],
                                  config['diffusion_model']['time_embedding_dim'],
                                  num_res_blocks=config['diffusion_model']['num_res_blocks'],
                                  kernel_size=config['diffusion_model']['kernel_size'])

    logging.info('\n\n\nDiffusion Model Training:')

    # Training of the Diffusion Model
    loss_history, mean_likelihood_history_time_steps, first_importance_sampling_epoch = (
        dm_trainer.train(diffusion_model, data_train))

    if writer:
        writer.close()

    diffusion_model = diffusion_model.cpu()

    # Save the model
    torch.save(diffusion_model.state_dict(), f'{save_path}/{dataset_name}_diffusion_model.pt')

    # Save the loss history and the mean log-likelihood history for all time steps
    torch.save(loss_history, f'{save_path}/{dataset_name}_loss_history.pt')
    torch.save(mean_likelihood_history_time_steps, f'{save_path}/{dataset_name}_mean_likelihood_history_time_steps.pt')

    # 4) GNN Training

    # Initialize the GNN trainer
    gnn_trainer = GNNTrainer(config['gnn_training'])

    if seed is not None:
        # Use a seed for reproducible initialization of the GCN
        torch.manual_seed(seed)

    # Initialize the model
    gnn = GCN(data.num_node_features, config['gnn']['hidden_channels'], config['gnn']['out_channels'])

    logging.info('\n\n\nGNN Training:')

    # Add the original node features to data_train
    data_train.x = data.x

    # Training of the GNN
    loss_history_train, loss_history_val = gnn_trainer.train(gnn, data_train)

    gnn = gnn.cpu()

    # Save the model
    torch.save(gnn.state_dict(), f'{save_path}/{dataset_name}_gnn.pt')

    # Save the training and validation loss histories of the GNN
    torch.save({'train_loss': loss_history_train, 'val_loss': loss_history_val},
               f'{save_path}/gnn_loss_history.pt')


if __name__ == '__main__':
    main()
