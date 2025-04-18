from typing import Any

import datasets
import torchvision
import numpy as np
import pandas as pd

from odfl.dataset_generation.utils import create_dirchlet_matrix


def dirchlet_from_hg(
    dataset: datasets.arrow_dataset.Dataset,
    agents: int,
    alpha: float = 0.5,
    seed: int = 42,
    shuffle: bool = True,
    list_format: bool = True,
    client_test_set_size: float = 0.2
    ):
    """Converts the centralised version of the dataset (HuggingFace Format) into
    a list (or dictionary) of a nodes."""
    # Dataset shuffling
    if shuffle:
        dataset = dataset.shuffle(seed = seed)
    # Creating dirchlet matrix for sampling (each vector is a draw from Dir(\alpha))
    d_matrix = create_dirchlet_matrix(
        alpha=alpha,
        size=agents,
        k=dataset.features['label'].num_classes
    )
    # Calculating average size of the sample 
    avg_size = int(np.floor(dataset.num_rows / agents))
    # Creating Pandas Dataframe of samples (useful for sampling)
    #pandas_df = dataset.to_pandas().drop('image', axis=1)
    pandas_df = dataset.to_pandas()
    if 'image' in pandas_df.columns:
        pandas_df = pandas_df.drop('image', axis=1)

    # Data format for storing data of all nodes.
    if list_format:
        nodes_data = []
    else:
        nodes_data = {}
    
    # Sampling for each client
    for agent in range(d_matrix.shape[0]):
        sampling_weights = pd.Series({label: p for (label, p) in zip(dataset.features['label'].names, d_matrix[agent])})
        pandas_df["weights"] = pandas_df['label'].apply(lambda x: sampling_weights[x])
        sample = pandas_df.sample(n = avg_size, weights='weights', random_state=seed)   
        sampled_data = dataset.select(sample.index)
        pandas_df.drop(sample.index, inplace=True)
        agent_data = sampled_data.train_test_split(test_size=client_test_set_size)
        if list_format:
            nodes_data.append([agent_data['train'], agent_data['test']])
        else:
            nodes_data[f"node_{agent}_train"] = agent_data['train']
            nodes_data[f"node_{agent}_test"] = agent_data['test']
    return nodes_data


def dirchlet_from_torch(
    dataset: torchvision.datasets,
    agents: int,
    alpha: float = 0.5,
    seed: int = 42,
    shuffle: bool = True,
    list_format: bool = True,
    client_test_set_size: float = 0.2
    ):
    """Converts the centralised version of the dataset (PyTorch Format) into
    a list (or dictionary) of a nodes."""
    pass
    # TODO


def dirchlet_from_custom(
    dataset: Any,
    agents: int,
    alpha: float = 0.5,
    seed: int = 42,
    shuffle: bool = True,
    list_format: bool = True,
    client_test_set_size: float = 0.2
    ):
    """Converts the centralised version of the dataset (Custom Format) into
    a list (or dictionary) of a nodes."""
    pass
    # TODO