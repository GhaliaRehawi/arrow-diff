import os
import random
import logging
import numpy as np
import torch

from time import time
from torch_geometric.data import Data
from torch_geometric.utils import degree
# from torch.utils.tensorboard import SummaryWriter

from arrow_diff.data import load_data
from arrow_diff.utils import read_config_file, initialize_logging
from arrow_diff.positional_encoding import positional_encoding
from arrow_diff.evaluation import evaluate_predicted_graph
from arrow_diff.unet import UNetAdapter

from arrow_diff.gcn import GCN
from arrow_diff.graph_generation import generate_graph


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

    if not os.path.isdir(f'{save_path}/graphs'):
        os.makedirs(f'{save_path}/graphs')

    # Initialize logging
    initialize_logging(save_path, experiment_name='logging')

    logging.info(f'Config:\n{config}')

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

    # Initialize the device
    device = torch.device('cuda' if config['graph_generation']['device'] == 'cuda' and torch.cuda.is_available()
                          else 'cpu')

    # 2) Load The Data

    # Load the single graph or its largest connected component from the dataset
    data = load_data(dataset_name, path=config['data']['path'], seed=seed)

    logging.info(f'\nGraph (LCC):\n{data}')

    num_nodes = data.num_nodes

    if config['node_feature_dim'] or dataset_name == 'SBM':
        # Use positional encodings as node features
        data.x = positional_encoding(torch.arange(num_nodes), config['node_feature_dim'])

    # 3) Load The Trained Diffusion Model

    # Initialize the model
    diffusion_model = UNetAdapter(config['diffusion_model']['hidden_channels'], num_nodes,
                                  config['diffusion_model']['node_embedding_dim'],
                                  config['dm_training']['num_diffusion_steps'],
                                  config['diffusion_model']['time_embedding_dim'],
                                  num_res_blocks=config['diffusion_model']['num_res_blocks'],
                                  kernel_size=config['diffusion_model']['kernel_size'])

    # Load the model parameters of the trained diffusion model
    diffusion_model.load_state_dict(torch.load(f'{save_path}/{dataset_name}_diffusion_model.pt'))

    # Set the diffusion model into evaluation mode
    diffusion_model.eval()

    # Move the diffusion model parameters to the specified device
    diffusion_model.to(device)

    # 4) Load The Trained GNN

    # Initialize the model
    gnn = GCN(data.num_node_features, config['gnn']['hidden_channels'], config['gnn']['out_channels'])

    # Load the model parameters of the trained gnn
    gnn.load_state_dict(torch.load(f'{save_path}/{dataset_name}_gnn.pt'))

    # Set the gnn into evaluation mode
    gnn.eval()

    # Move the gnn parameters to the specified device
    gnn.to(device)

    # 5) Graph Generation

    logging.info(f'\n\n\nGraph Generation:')

    num_steps = config['graph_generation']['num_steps']

    # Compute the node degrees of the nodes in the original graph
    deg_gt = degree(data.edge_index[0], num_nodes).to(device)

    # Move the node features to the specified device
    x = data.x.to(device)

    metrics = {
        'edge_overlap': [],
        'degree_assort': [],
        'avg_clustering_coef': [],
        'global_clustering_coef': [],
        'power_law_exp': [],
        'num_triangles': [],
        'max_degree': [],
        'time': []
    }

    start_time = time()

    for i in range(config['graph_generation']['num_samples'] + 1):
        logging.info(f'\nGenerating graph {i}:')

        start_time_i = time()

        edge_index_pred = generate_graph(diffusion_model, config['dm_training']['batch_size'],
                                         config['dm_training']['random_walks']['walk_length'], gnn, x, deg_gt,
                                         num_steps, device=device, seed=i * seed)

        time_i = time() - start_time_i

        edge_index_pred = edge_index_pred.cpu()

        logging.info(f'Graph generation took {time_i:.3f} seconds')
        logging.info(f'Number of edges in predicted graph {i}: {edge_index_pred.size(1) // 2}')

        (eo, degree_assort, avg_clustering_coef, global_clustering_coef, power_law_exp, num_triangles, max_degree,
         connected_component_sizes) = evaluate_predicted_graph(edge_index_pred, data.edge_index, num_nodes)

        if i > 0:
            logging.info('\nMetrics:')
            logging.info(f'Edge overlap: {eo}')
            logging.info(f'Degree Assortativity: {degree_assort}')
            logging.info(f'Average Clustering Coefficient: {avg_clustering_coef}')
            logging.info(f'Global Clustering Coefficient: {global_clustering_coef}')
            logging.info(f'Power Law Exponent: {power_law_exp}')
            logging.info(f'Number of Triangles: {num_triangles}')
            logging.info(f'Maximum Node Degree: {max_degree}')
            logging.info(f'Time: {time_i:.3f}')
            logging.info(f'{len(connected_component_sizes)} Connected Components with Sizes: '
                         f'{connected_component_sizes}')

            metrics['edge_overlap'].append(eo)
            metrics['degree_assort'].append(degree_assort)
            metrics['avg_clustering_coef'].append(avg_clustering_coef)
            metrics['global_clustering_coef'].append(global_clustering_coef)
            metrics['power_law_exp'].append(power_law_exp)
            metrics['num_triangles'].append(num_triangles)
            metrics['max_degree'].append(max_degree)
            metrics['time'].append(time_i)

            data_pred = Data(edge_index=edge_index_pred, num_nodes=num_nodes)

            # Save the predicted graph
            torch.save(data_pred, f'{save_path}/graphs/graph_{i}.pt')

    logging.info(f'Generation of {config["graph_generation"]["num_samples"]} graphs took {time() - start_time:.3f} '
                 f'seconds')

    torch.save(metrics, f'{save_path}/metrics_per_graph.pt')

    metrics_averaged = dict()

    for metric in metrics:
        std, mean = torch.std_mean(torch.as_tensor(metrics[metric]).float())
        metrics_averaged[metric] = (mean.item(), std.item())

    torch.save(metrics_averaged, f'{save_path}/metrics_averaged.pt')


if __name__ == '__main__':
    main()
