import os
from datasets import load_dataset
from odfl.dataset_generation.dataset_generator import dirchlet_from_hg
from odfl.dataset_generation.utils import save_custom_dataset

def generate_agnews(
    agents: int,
    alpha: float,
    seed: int = 42,
    shuffle: bool = True,
    client_test_size: float = 0.2
):
    # Create required folders if they don't exist
    os.makedirs(os.path.join(os.getcwd(), 'centralised'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'decentralised'), exist_ok=True)

    # Load AG News dataset from Hugging Face
    dataset = load_dataset('ag_news')
    
    # Save full dataset to 'centralised' folder
    dataset.save_to_disk(os.path.join(os.getcwd(), 'centralised', 'AGNews'))

    # Extract orchestrator and node datasets
    orchestrator_data = dataset['test']
    nodes_data = dataset['train']

    # Perform Dirichlet-based non-IID split
    data_split = dirchlet_from_hg(
        dataset=nodes_data,
        agents=agents,
        alpha=alpha,
        seed=seed,
        shuffle=shuffle,
        client_test_set_size=client_test_size,
        list_format=False
    )

    # Assign the test set to orchestrator
    data_split['orchestrator_dataset'] = orchestrator_data

    # Save the custom federated dataset to 'decentralised' folder
    save_custom_dataset(
        path=os.path.join(os.getcwd(), 'decentralised', 'AGNews'),
        dataset=data_split
    )

if __name__ == "__main__":
    generate_agnews(
        agents=3,     # Number of clients
        alpha=0.5     # Dirichlet alpha (non-IID degree)
    )
