import os

import datasets
import numpy as np


def create_dirchlet_matrix(
    alpha: int | float,
    size: int,
    k: int
    ):
    generator = np.random.default_rng()
    alpha = np.full(k, alpha)
    s = generator.dirichlet(alpha, size)
    return s


def save_custom_dataset(
    path: str,
    dataset: dict
    ):
    dataset = datasets.DatasetDict(dataset)
    dataset.save_to_disk(path)


def convert_to_hg(
    dataset: list[dict]
):
    dataset = [{'image': dataset[index][0], 'label': int(dataset[index][1][0])} for index in range(len(dataset))]
    dataset = datasets.Dataset.from_list(dataset)
    return dataset

# def save_blueprint(
#     path: str,
#     dataset: dict[datasets.arrow_dataset.Dataset],
#     blueprint_name: str,
#     number_of_clients: int
# ):
#     name = os.path.join(path, (blueprint_name + ".csv"))
#     with open(name, "w+") as csv_file:
#         header = ['client_id', 'partition', 'total_samples']
#         labels = dataset[f"node_0_train"].features['label'].names
#         header.extend(labels)
#         csv_file.write(",".join(header) + '\n')
        
#         translation = False
#         try:
#             labels = [int(label) for label in labels]
#         except ValueError:
#             translation = True
#             labels_translation = {integer: label for (integer, label) in zip(range(len(labels)), labels)}
#             labels = [value for value in labels_translation.keys()]
        
#         # WRITE ORCHESTRATOR
#         row = ['orchestrator', 'central_test_set', str(len(dataset['orchestrator_data']))]
#         for label in labels:
#             row.append(str(len(dataset['orchestrator_data'].filter(lambda inst: inst['label'] == label))))
#         csv_file.write(",".join(row) + '\n')
        
        
#         # WRITE CLIENTS
#         for clients in range(number_of_clients):
#             pass
        
        
        
#         for client, data in enumerate(nodes_data):
#             row = [str(client), 'train_set', str(len(data[0]))]
#             for label in labels:
#                 row.append(str(len(data[0].filter(lambda inst: inst['label'] == label))))
#             csv_file.write(",".join(row) + '\n')

#             row = [str(client), 'test_set', str(len(data[1]))]
#             for label in labels:
#                 row.append(str(len(data[1].filter(lambda inst: inst['label'] == label))))
#             csv_file.write(",".join(row) + '\n')
        
#         #Optional: write labels (if translated)
#         if translation:
#             name = os.path.join(path, (blueprine_name + "translation" + ".txt"))
#             with open(name, "w+") as txt_file:
#                 for integer, label in labels_translation.items():
#                     txt_file.write(f"{str(integer)}: {label}")