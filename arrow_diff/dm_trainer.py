import logging
import torch

from torch import FloatTensor
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch.optim import Adam
from torch.nn import Module
# from torch.utils.tensorboard import SummaryWriter
from time import time
from typing import List, Optional, Tuple

from arrow_diff.time_step_sampler import TimeStepSampler

from arrow_diff.random_walks import sample_random_walks


class DiffusionModelTrainer:
    """
    Trainer for diffusion model training.
    """
    # def __init__(self, config: dict, writer: Optional[SummaryWriter] = None) -> None:
    def __init__(self, config: dict, writer=None) -> None:
        """
        Initialization of DiffusionModelTrainer.

        Args:
            config: dict
                Configuration with training parameters.
            writer: SummaryWriter (optional, default: None)
                SummaryWriter to track the loss in real time.
        """
        self.config = config

        self.writer = writer

        # Initialize the device
        self.device = torch.device('cuda' if config['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')

        # Initialize generators on CPU and GPU (if specified) for reproducibility
        self.generator_cpu = None if self.config['seed'] is None else torch.Generator(device='cpu')
        self.generator_gpu = None if self.config['seed'] is None else torch.Generator(device=self.device)

    def train(self, model: Module, data: Data) -> Tuple[List[float], FloatTensor, Optional[int]]:
        """
        Performs the model training.

        Args:
            model: torch.nn.Module
                Model that predicts the probabilities p(x_tm1 | x_t) of the reverse diffusion process.
            data: torch_geometric.data.Data
                PyTorch Geometric Data object that represents a single graph to train the model on.

        Returns:
            loss_history: List[float]
                History of loss values.
            mean_likelihood_history_time_steps: torch.FloatTensor, shape: (num_epochs,)
                Gives the mean likelihood for each epoch and time step.
            first_importance_sampling_epoch: int (optional)
                Gives the first epoch with time importance sampling if time importance sampling is used.
        """
        # Set the model into training mode
        model.train()

        # Move the model parameters to the specified device
        model.to(self.device)

        # Initialize the optimizer
        optimizer = Adam(model.parameters(), **self.config['optimizer']['kwargs'])

        # Move the data to the specified device
        data = data.to(self.device)

        # Convert the data to a SparseTensor to obtain the rowptr and col vectors for the random_walk() function
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                           sparse_sizes=(data.num_nodes, data.num_nodes))
        rowptr, col, _ = adj.csr()

        # Obtain the number of nodes
        num_nodes = data.num_nodes

        num_epochs = self.config['num_epochs']

        # Walk length is given in terms of edges
        walk_length = self.config['random_walks']['walk_length']

        seed = self.config['seed']

        # Initialize the time step sampler
        time_step_sampler = TimeStepSampler(self.config['num_diffusion_steps'], self.config['num_history_values'],
                                            mode=self.config['time_step_sampling'], device=self.device)

        # Track the epoch where time step sampling is used the first time
        first_importance_sampling_epoch = None

        # Save the loss history in a list
        loss_history = []

        # History of likelihood for each time step
        mean_likelihood_history_time_steps = torch.zeros(num_epochs, self.config['num_diffusion_steps'])

        training_start_time = time()

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time()

            if self.generator_gpu is not None:
                # Seed the generator for reproducibility
                self.generator_gpu.manual_seed(epoch * seed)

            # Sample random walks
            random_walks = sample_random_walks(rowptr, col, self.config['batch_size'],
                                               self.config['random_walks']['walk_length'] - 1,
                                               self.config['random_walks']['p'], self.config['random_walks']['q'],
                                               generator=self.generator_gpu,
                                               seed=None if seed is None else epoch * seed)

            if self.generator_gpu is not None:
                # Seed the generator for reproducibility
                self.generator_gpu.manual_seed(epoch * seed)

            # Obtain sampled time steps together with the probability values with which they were sampled
            time_steps, time_step_probs = time_step_sampler.sample_time_steps(self.config['batch_size'],
                                                                              generator=self.generator_gpu)

            # Sort time_steps and time_step_probs to make indexing of the dense prob_x_tm1_pred faster
            time_steps, idxs = torch.sort(time_steps)
            # TODO: Use this if time importance sampling is used
            # time_step_probs = time_step_probs[idxs]

            # Generate one permutation per random walk
            permutations = []

            for i in range(1, self.config['batch_size'] + 1):
                if self.generator_gpu is not None:
                    # Seed the generator for reproducibility
                    self.generator_gpu.manual_seed(i * epoch * seed)

                # Generate a permutation of size walk_length
                permutations.append(torch.randperm(walk_length, generator=self.generator_gpu, device=self.device))

            # Stack the permutations along the batch dimension to a single torch.LongTensor/torch.cuda.LongTensor
            perm = torch.stack(permutations, dim=0)

            # Compute the mask
            mask = perm < time_steps.unsqueeze(1)

            # Compute the diffusion by replacing nodes in the random walks for which sigma(< time_steps) with the mask
            nodes = torch.where(mask, random_walks, num_nodes)

            # Zero the gradients of the model parameters
            optimizer.zero_grad()

            # Get the model predictions (log-probabilities, shape: (batch_size, walk_length, num_nodes))
            # for all k \in sigma(>= t), independent of the element in the batch,
            # shape: (sum_i |sigma_i(>= t)|, num_nodes), where i is the index of a random walk in the batch
            log_prob = model(nodes, time_steps)[~mask]

            # This gives the x_k, where k \in sigma(>= t), independent of the element in the batch,
            # shape: (sum_i |sigma_i(>= t)|,), where i is the index of a random walk in the batch
            x_k = random_walks[~mask]

            # This gives the log-probabilities for all x_k, independent of the element in the batch,
            # shape: (sum_i |sigma_i(>= t)|,), where i is the index of a random walk in the batch
            log_prob = log_prob[torch.arange(log_prob.size(0)), x_k]

            # This gives indices of the elements (random walks) in the batch for x_k,
            # shape: (sum_i |sigma_i(>= t)|,), where i is the index of a random walk in the batch
            element = torch.where(~mask)[0]

            # This gives sum_{k \in sigma(>= t)} log(p(x_k | x_{sigma(< t)})) for every random walk in the batch
            log_prob = scatter(log_prob, element)

            # Compute the weight for the loss: 1 / (D - t + 1), where D is the walk_length
            loss_weight = 1 / (walk_length - (time_steps + 1) + 1)

            # Compute the weighted loss value for each random walk in the batch
            loss_values = loss_weight * log_prob

            # Compute the negative mean of D * loss_values to get the upper bound, where D is the walk_length
            loss = -torch.mean(walk_length * loss_values)

            # TODO: Use this if time importance sampling is used
            # Compute the resulting loss which is the mean of the loss_values for time_steps rescaled by time_step_probs
            #loss = -torch.mean(walk_length * loss_values / time_step_probs)

            # Compute the gradients
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # TODO: Use this if time importance sampling is used
            # if self.generator_cpu is not None:
            #     # Seed the generator for reproducibility
            #     self.generator_cpu.manual_seed(epoch * seed)

            # Update the history of time step-dependent loss values of the time step sampler
            # time_step_sampler.update_loss_history(time_steps.cpu(), loss_values.detach().cpu(),
            #                                       generator=self.generator_cpu)

            # Store the loss (mean of likelihood values for sampled time steps) in the loss history
            loss_history.append(loss.item() / walk_length)
            # loss_history.append(torch.mean(loss_values.detach()).item())

            mean_likelihood_history_time_steps[epoch - 1] = scatter(loss_values.detach().cpu(), time_steps.cpu(),
                                                                    dim_size=self.config['num_diffusion_steps'],
                                                                    reduce='mean')

            if first_importance_sampling_epoch is None and time_step_sampler.mode == 'importance' and \
                    torch.all(time_step_sampler.time_step_counts >= time_step_sampler.num_history_values):
                first_importance_sampling_epoch = epoch

            if epoch == 1 or epoch % int(0.1 * num_epochs) == 0:
                logging.info(f'Epoch: {epoch}/{num_epochs}\t-\tTime per epoch: {time() - epoch_start_time:.3f} seconds'
                             f'\t-\tLoss: {loss.item():.3f}')

            if self.writer:
                self.writer.add_scalar('Loss/Weighted', loss.item(), global_step=epoch)
                self.writer.add_scalar('Loss/Unweighted', torch.mean(loss_values.detach()).item(), global_step=epoch)

            # Stopping criterion
            if epoch >= 1000 and sum(loss_history[-500:]) / 500 > sum(loss_history[-1000:-500]) / 500:
                logging.info(f'Early stopping at epoch {epoch}')
                break

        if time_step_sampler.mode == 'importance' and first_importance_sampling_epoch is not None:
            logging.info(f'Time importance sampling started in epoch {first_importance_sampling_epoch}.')

        logging.info(f'Training finished after {time() - training_start_time:.3f} seconds')

        return loss_history, mean_likelihood_history_time_steps, first_importance_sampling_epoch
