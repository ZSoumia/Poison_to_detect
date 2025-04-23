import os

import medmnist
import datasets
from datasets import load_dataset

from odfl.dataset_generation.dataset_generator import dirchlet_from_custom
from odfl.dataset_generation.utils import save_custom_dataset
from odfl.dataset_generation.utils import convert_to_hg


def generate_mnist(
    agents: int,
    alpha: float,
    seed: int = 42,
    shuffle: bool = True,
    client_test_size: float = 0.2
):
    # Dataset Source: https://github.com/MedMNIST/MedMNIST
    os.makedirs(os.path.join(os.getcwd(), 'centralised'))
    os.makedirs(os.path.join(os.getcwd(), 'decentralised'))
    # Loading the dataset
    pathMNIST_train = medmnist.PathMNIST(split='train', size=28, download=True)
    pathMNIST_test = medmnist.PathMNIST(split='test', size=28, download=True)
    pathMNIST_val = medmnist.PathMNIST(split='val', size=28, download=True)
    # Joining the val and train dataset together
    centralised_dataset = pathMNIST_train + pathMNIST_val + pathMNIST_test
    # Converting the datasets to HG format
    centralised_dataset = convert_to_hg(centralised_dataset)
    centralised_dataset.save_to_disk(os.path.join(os.getcwd(), 'centralised', 'PathMNIST'))
    
    orchestrator_data = pathMNIST_test + pathMNIST_val
    orchestrator_data = convert_to_hg(orchestrator_data)
    pathMNIST_train = convert_to_hg(pathMNIST_train)
    data_split = dirchlet_from_custom(
        dataset=pathMNIST_train,
        agents=agents,
        alpha=alpha,
        seed=seed,
        shuffle=shuffle,
        client_test_set_size=client_test_size,
        list_format=False,
        number_of_classes=9,
        labels = [str(label) for label in range(9)]
    )
    data_split['orchestrator_dataset'] = orchestrator_data
    save_custom_dataset(
        path = (os.path.join(os.getcwd(), 'decentralised', 'PathMNIST')),
        dataset = data_split
        )
    

if __name__ == "__main__":
    generate_mnist(
        agents=3,
        alpha=0.2
    )