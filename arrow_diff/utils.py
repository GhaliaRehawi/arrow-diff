import sys
import yaml
import logging


def read_config_file(path: str) -> dict:
    """
    Reads a .yaml config file into a dict.

    Args:
        path: str
            Path to the config file.

    Returns:
        config: dict
            Parameters specified in the config file.
    """
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError:
            print('Could not load config file!')

    return config


def save_config_to_file(config: dict, path: str) -> None:
    """
    Reads a .yaml config file into a dict.

    Args:
        config: dict
            Parameters of the config.
        path: str
            Path to the config file.
    """
    with open(f'{path}/config.yaml', 'w') as stream:
        try:
            yaml.safe_dump(config, stream=stream)
        except yaml.YAMLError:
            print('Could not save the config to the specified file!')


def initialize_logging(stream: str, experiment_name: str = 'experiment') -> None:
    """
    Initializes the logging.

    Args:
        stream: str
            Either 'stdout' to print the logging to stdout or a filename.
        experiment_name: str (optional, default: 'experiment')
            Name of the experiment, which will be used as the filename for logging if stream is not 'stdout'.
    """
    if stream == 'stdout':
        logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(filename=f'{stream}/{experiment_name}.log', format='%(message)s', level=logging.INFO)
