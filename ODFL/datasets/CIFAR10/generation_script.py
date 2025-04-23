import os

from datasets import load_dataset

from odfl.dataset_generation.dataset_generator import dirchlet_from_hg
from odfl.dataset_generation.utils import save_custom_dataset, save_blueprint

def generate_mnist(
    agents: int,
    alpha: float,
    seed: int = 42,
    shuffle: bool = True,
    client_test_size: float = 0.2
):
    # Dataset Source: uoft-cs/cifar10
    os.makedirs(os.path.join(os.getcwd(), 'centralised'))
    os.makedirs(os.path.join(os.getcwd(), 'decentralised'))
    dataset = load_dataset('uoft-cs/cifar10')
    dataset = dataset.rename_column('img', 'image')
    dataset.save_to_disk(os.path.join(os.getcwd(), 'centralised', 'CIFAR10'))
    orchestrator_data = load_dataset('uoft-cs/cifar10', split='test') # Orchestrator Data
    orchestrator_data = orchestrator_data.rename_column('img', 'image')
    nodes_data = load_dataset('uoft-cs/cifar10', split='train') # Nodes Data (to be split)
    nodes_data = nodes_data.rename_column('img', 'image')
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
        path = (os.path.join(os.getcwd(), 'decentralised', 'CIFAR10')),
        dataset = data_split
        )
    save_blueprint(
        path = (os.path.join(os.getcwd(), 'decentralised', 'CIFAR10')),
        dataset = data_split,
        blueprint_name='CIFAR10_SPLIT',
        number_of_clients=agents
    )

if __name__ == "__main__":
    generate_mnist(
        agents=3,
        alpha=0.2
    )