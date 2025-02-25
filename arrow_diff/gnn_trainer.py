import math
import logging
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import Module, BCEWithLogitsLoss
from time import time
from typing import List, Tuple

from arrow_diff.graph_dataset import GraphPerturbationDataset


class GNNTrainer:
    """
    Trainer for GNN training.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialization of GNNTrainer.

        Args:
            config: dict
                Configuration with training parameters.
        """
        self.config = config

        # Initialize the device
        self.device = torch.device('cuda' if config['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')

        # Initialize generators on CPU and GPU (if specified) for reproducibility
        self.generator_cpu = None if self.config['seed'] is None else torch.Generator(device='cpu')
        self.generator_gpu = None if self.config['seed'] is None else torch.Generator(device=self.device)

    def train(self, model: Module, data: Data) -> Tuple[List[float], List[float]]:
        # Move the model parameters to the specified device
        model.to(self.device)

        # Initialize the optimizer
        optimizer = Adam(model.parameters(), **self.config['optimizer']['kwargs'])

        num_graphs = self.config['num_graphs']
        batch_size = self.config['batch_size']
        edge_dropout = self.config['edge_dropout']
        fake_edge_ratios = tuple(self.config['fake_edge_ratios'])
        num_epochs = self.config['num_epochs']
        seed = self.config['seed']

        dataset = GraphPerturbationDataset(data.x.cpu(), data.edge_index.cpu(), num_graphs, edge_dropout,
                                           fake_edge_ratios, seed=seed)

        if self.generator_cpu:
            self.generator_cpu.manual_seed(self.config['seed'])

        dataset_train, dataset_val = random_split(dataset, [0.6, 0.4], generator=self.generator_cpu)

        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=self.generator_cpu)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        expected_num_valid_edges = int((1 - edge_dropout) * data.edge_index.size(1))
        expected_num_fake_edges = int(sum(fake_edge_ratios) / 2 * expected_num_valid_edges)
        bce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([expected_num_fake_edges / expected_num_valid_edges],
                                                             device=self.device))

        loss_history_train = []
        loss_history_val = []

        best_val_loss = math.inf
        patience = self.config['patience']

        training_start_time = time()

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time()

            # Set the model to training mode
            model.train()

            losses = []

            if self.generator_cpu:
                self.generator_cpu.manual_seed(epoch * seed)

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                node_embeddings = model(batch.x, batch.edge_index)
                row, col = batch.edge_index
                y_pred = torch.sum(node_embeddings[row] * node_embeddings[col], dim=1)
                loss = bce_loss(y_pred, batch.y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            loss_history_train.append(sum(losses) / len(losses))

            # Set the model to evaluation mode
            model.eval()

            losses = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    node_embeddings = model(batch.x, batch.edge_index)
                    row, col = batch.edge_index
                    y_pred = torch.sum(node_embeddings[row] * node_embeddings[col], dim=1)
                    loss = bce_loss(y_pred, batch.y)
                    losses.append(loss.item())

            loss_history_val.append(sum(losses) / len(losses))

            if loss_history_val[-1] < best_val_loss:
                best_val_loss = loss_history_val[-1]
                patience = self.config['patience']
            else:
                patience -= 1

            logging.info(f'Epoch: {epoch} finished after {time() - epoch_start_time:.3f} seconds '
                         f'- Average train loss: {loss_history_train[-1]:.6f} '
                         f'- Average validation loss: {loss_history_val[-1]:.6f}')

            if patience == 0:
                logging.info(f'Early stopping')
                break

        logging.info(f'Training finished after {time() - training_start_time:.3f} seconds')

        return loss_history_train, loss_history_val
