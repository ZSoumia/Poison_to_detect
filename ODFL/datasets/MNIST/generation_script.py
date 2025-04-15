import os

import datasets
from datasets import load_dataset

from odfl.dataset_generation.dataset_generator import dirchlet_from_hg
from odfl.dataset_generation.utils import save_custom_dataset

def generate_mnist(
    agents: int,
    alpha: float,
    seed: int = 42,
    shuffle: bool = True,
    client_test_size: float = 0.2
):
    # Dataset Source: https://huggingface.co/datasets/ylecun/mnist
    os.makedirs(os.path.join(os.getcwd(), 'centralised'))
    os.makedirs(os.path.join(os.getcwd(), 'decentralised'))
    dataset = load_dataset('ylecun/mnist')
    dataset.save_to_disk(os.path.join(os.getcwd(), 'centralised', 'MNIST'))
    orchestrator_data = load_dataset('ylecun/mnist', split='test') # Orchestrator Data
    nodes_data = load_dataset('ylecun/mnist', split='train') # Nodes Data (to be split)
    data_split = dirchlet_from_hg(
        dataset=nodes_data,
        agents=agents,
        alpha=alpha,
        seed=seed,
        shuffle=shuffle,
        client_test_set_size=client_test_size,
        list_format=False
    )
    data_split['orchestrator_dataset'] = orchestrator_data
    save_custom_dataset(
        path = (os.path.join(os.getcwd(), 'decentralised', 'MNIST')),
        dataset = data_split
        )


if __name__ == "__main__":
    generate_mnist(
        agents=3,
        alpha=0.2
    )