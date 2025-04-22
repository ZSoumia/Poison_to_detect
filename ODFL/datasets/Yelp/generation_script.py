import os
from datasets import load_dataset
from odfl.dataset_generation.dataset_generator import dirchlet_from_hg
from odfl.dataset_generation.utils import save_custom_dataset

def generate_yelp(
    agents: int,
    alpha: float,
    seed: int = 42,
    shuffle: bool = True,
    client_test_size: float = 0.2
):
    # Create output folders
    os.makedirs(os.path.join(os.getcwd(), 'centralised'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'decentralised'), exist_ok=True)

    # Load Yelp Review Full dataset
    dataset = load_dataset('yelp_review_full')

    # Save entire dataset to the centralised folder
    dataset.save_to_disk(os.path.join(os.getcwd(), 'centralised', 'YelpReviewFull'))

    # Extract train and test splits
    orchestrator_data = dataset['test']    # Will be used by orchestrator
    nodes_data = dataset['train']          # Used for client splitting

    # Create client-specific datasets with Dirichlet-based non-IID split
    data_split = dirchlet_from_hg(
        dataset=nodes_data,
        agents=agents,
        alpha=alpha,
        seed=seed,
        shuffle=shuffle,
        client_test_set_size=client_test_size,
        list_format=False
    )

    # Attach test set for orchestrator
    data_split['orchestrator_dataset'] = orchestrator_data

    # Save to decentralised folder
    save_custom_dataset(
        path=os.path.join(os.getcwd(), 'decentralised', 'YelpReviewFull'),
        dataset=data_split
    )

if __name__ == "__main__":
    generate_yelp(
        agents=5,     # Number of clients
        alpha=0.3     # Degree of non-IID-ness
    )
