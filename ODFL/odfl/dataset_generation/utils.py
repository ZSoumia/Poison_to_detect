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