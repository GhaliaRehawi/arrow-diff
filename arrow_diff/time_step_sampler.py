import torch

from typing import Optional, Tuple, TypeVar


# Create TypeVars for FloatTensor and LongTensor
FloatTensor = TypeVar('FloatTensor', torch.FloatTensor, torch.cuda.FloatTensor)
LongTensor = TypeVar('LongTensor', torch.LongTensor, torch.cuda.LongTensor)


class TimeStepSampler:
    """
    TimeStepSampler to sample time steps either uniformly or with importance sampling (see [1]).

    [1] A. Nichol and P. Dhariwal. "Improved Denoising Diffusion Probabilistic Models".
        International Conference of Machine Learning, PMLR 139, 2021.
    """
    def __init__(self, num_diffusion_steps: int, num_history_values: int, mode: str = 'uniform',
                 device: Optional[torch.device] = None) -> None:
        """
        Initialization of TimeStepSampler.

        Args:
            num_diffusion_steps: int
                Number of diffusion time steps.
            num_history_values: int
                Number of loss values to store per time step.
            mode: str (optional, default: 'uniform')
                Sampling mode: Either 'uniform' or 'importance'.
            device: torch.device (optional, default: None)
                Device on which to return the torch.Tensors.
        """
        self.num_diffusion_steps = num_diffusion_steps
        self.num_history_values = num_history_values
        self.mode = mode
        self.device = device

        if mode == 'importance':
            # A counter for each time step
            self.time_step_counts = torch.zeros(num_diffusion_steps, dtype=torch.long)

            # A matrix that stores the last 10 squared loss values for each time step
            self.squared_loss_history = torch.zeros(num_diffusion_steps, num_history_values)

    def uniform_sampling(self, num_samples: int,
                         generator: Optional[torch.Generator] = None) -> Tuple[LongTensor, FloatTensor]:
        """
        Uniform time step sampling.

        Args:
            num_samples: int
                Number of time steps to sample.
            generator: torch.Generator (optional, default: None)
                Generator for reproducible time step sampling.

        Returns:
            time_steps: torch.LongTensor, shape: (num_samples,)
                Sampled time steps.
            probs: torch.FloatTensor, shape: (num_samples,)
                Probabilities with which the time steps where sampled.
        """
        # Sample time steps
        time_steps = torch.randint(self.num_diffusion_steps, (num_samples,), generator=generator, device=self.device)

        # Compute uniform probabilities
        probs = torch.ones(num_samples, device=self.device) / self.num_diffusion_steps

        if self.mode == 'importance':
            # Update self.time_step_counts
            self.time_step_counts.scatter_add_(0, time_steps.cpu(), torch.ones(num_samples, dtype=torch.long))

        return time_steps, probs

    def importance_sampling(self, num_samples: int,
                            generator: Optional[torch.Generator] = None) -> Tuple[LongTensor, FloatTensor]:
        """
        Importance time step sampling [1].

        [1] A. Nichol and P. Dhariwal. "Improved Denoising Diffusion Probabilistic Models".
        International Conference of Machine Learning, PMLR 139, 2021.

        Args:
            num_samples: int
                Number of time steps to sample.
            generator: torch.Generator (optional, default: None)
                Generator for reproducible time step sampling.

        Returns:
            time_steps: torch.LongTensor, shape: (num_samples,)
                Sampled time steps.
            probs: torch.FloatTensor, shape: (num_samples,)
                Probabilities with which the time steps where sampled.
        """
        # Compute the probabilities for time step sampling
        probs = torch.sqrt(torch.mean(self.squared_loss_history, dim=1))
        probs /= torch.sum(probs)

        # Sample the time steps based on the computed probabilities
        time_steps = torch.multinomial(probs.to(self.device), num_samples, replacement=True, generator=generator)

        # Select the probabilities of the sampled time steps
        probs = probs[time_steps.cpu()].to(self.device)

        return time_steps, probs

    def sample_time_steps(self, num_samples: int,
                          generator: Optional[torch.Generator] = None) -> Tuple[LongTensor, FloatTensor]:
        """
        Sample time steps based on self.mode.

        Args:
            num_samples: int
                Number of time steps to sample.
            generator: torch.Generator (optional, default: None)
                Generator for reproducible time step sampling.

        Returns:
            time_steps: torch.LongTensor, shape: (num_samples,)
                Sampled time steps.
            probs: torch.FloatTensor, shape: (num_samples,)
                Probabilities with which the time steps where sampled.
        """
        if self.mode == 'uniform' \
                or self.mode == 'importance' and torch.any(self.time_step_counts < self.num_history_values):
            return self.uniform_sampling(num_samples, generator=generator)
        elif self.mode == 'importance':
            return self.importance_sampling(num_samples, generator=generator)
        else:
            raise ValueError(f'Mode {self.mode} is not implemented!')

    def update_loss_history(self, time_steps: LongTensor, loss_values: FloatTensor,
                            generator: Optional[torch.Generator] = None) -> None:
        """
        Update self.squared_loss_history.

        Args:
            time_steps: torch.LongTensor, shape: (num_samples,)
                Sampled time steps.
            loss_values: torch.FloatTensor, shape: (num_samples,)
                A loss value for each time step.
            generator: torch.Generator (optional, default: None)
                Generator for reproducible sampling of loss values if the number of
                loss values per time step is greater than self.num_history_values.
        """
        # Square the loss values
        squared_loss_values = torch.square(loss_values)

        unique_time_steps, unique_time_step_counts = torch.unique(time_steps, return_counts=True)

        for time_step, time_step_count in zip(unique_time_steps.tolist(), unique_time_step_counts.tolist()):
            if time_step_count <= self.num_history_values:
                # If the number of loss values for time step time_step is less or equal than self.num_history_values,
                # store all new loss values
                tmp = self.squared_loss_history[time_step, time_step_count:].clone()
                self.squared_loss_history[time_step, :-time_step_count] = tmp
                self.squared_loss_history[time_step, -time_step_count:] = squared_loss_values[time_steps == time_step]
            else:
                # If the number of loss values for time step time_step is greater than self.num_history_values,
                # sample self.num_history_values loss values and store them
                if generator:
                    # Seed the generator for reproducible sampling
                    generator.manual_seed(time_step_count)

                # Uniformly sample self.num_history_values loss values
                idxs = torch.multinomial(torch.ones(time_step_count, device=generator.device), self.num_history_values,
                                         generator=generator).cpu()

                # Store the sampled loss values for time step time_step
                self.squared_loss_history[time_step] = squared_loss_values[time_steps == time_step][idxs]
