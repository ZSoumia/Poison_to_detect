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
    # Dataset Source: zh-plus/tiny-imagenet
    os.makedirs(os.path.join(os.getcwd(), 'centralised'))
    os.makedirs(os.path.join(os.getcwd(), 'decentralised'))
    dataset = load_dataset('zh-plus/tiny-imagenet')
    dataset.save_to_disk(os.path.join(os.getcwd(), 'centralised', 'TinyImageNet'))
    orchestrator_data = load_dataset('zh-plus/tiny-imagenet', split='valid') # Orchestrator Data
    nodes_data = load_dataset('zh-plus/tiny-imagenet', split='train') # Nodes Data (to be split)
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
        path = (os.path.join(os.getcwd(), 'decentralised', 'TinyImageNet')),
        dataset = data_split
        )
    save_blueprint(
        path = (os.path.join(os.getcwd(), 'decentralised', 'TinyImageNet')),
        dataset = data_split,
        blueprint_name='TinyImageNet_SPLIT',
        number_of_clients=agents
    )    


if __name__ == "__main__":
    generate_mnist(
        agents=3,
        alpha=0.2
    )