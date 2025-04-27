import os
import csv
import time
import json
import shutil
import random

import datasets
import torch
import numpy as np

from odfl.centralised_training.training_utils import train_loop, test_loop, transform_func


WIDTH = shutil.get_terminal_size().columns
DATASET_PATH = ''
RESULTS_PATH = ''
# RANDOM SEEDS
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
def launch_centralised_tinyimagenet(
    lr: float,
    iterations: float,
    batch: int,
    device: torch.device,
    nesterov: bool = False,
    momentum: float = 0,
    dampening: float = 0,
    weight_decay: float = 0,
    scheduler: bool = False,
    scheduler_rounds: list = None,
    scheduler_gamma: float = None
):
    """
    Train a centralized TINYIMAGENET model with configurable optimizer and optional learning rate scheduler.

    This function sets up and runs a training loop for a centralized (non-federated) TINYIMAGENET model.
    It configures a JSON file with the provided hyperparameters, performs training on the specified device, 
    and saves outputs (results, logs, model checkpoints) into an organized directory.

    Args:
        lr (float): Learning rate for the optimizer.
        iterations (float): Number of training iterations (epochs or steps).
        batch (int): Batch size used for training.
        device (torch.device): Computation device (e.g., "cpu" or "cuda").
        nesterov (bool, optional): Whether to use Nesterov momentum. Defaults to False.
        momentum (float, optional): Momentum factor for the optimizer. Defaults to 0.
        dampening (float, optional): Dampening applied to momentum. Defaults to 0.
        weight_decay (float, optional): Weight decay (L2 regularization penalty). Defaults to 0.
        scheduler (bool, optional): Whether to use a learning rate scheduler. Defaults to False.
        scheduler_rounds (list, optional): List of training rounds (epochs) at which to adjust the learning rate. 
                                           Required if `scheduler` is True.
        scheduler_betas (float): A multiplicative factor (gamma) for adjusting the learning rate at specified rounds.
                                          Required if `scheduler` is True.

    Returns:
        None

    Side Effects:
        - Saves a JSON configuration file with all training hyperparameters.
        - Saves model checkpoints, training logs, and results into an output directory.

    Raises:
        ValueError: If `scheduler` is True but `scheduler_rounds` or `scheduler_betas` are not provided.
    """
    # ---------------------- Phase I: Create the target directory, define the metrics files ----------------------
    print(('-'*WIDTH).center(WIDTH))
    print('CREATING RELEVANT DIRECTORIES'.center(WIDTH))
    print(('-'*WIDTH).center(WIDTH))
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    dataset_directory = os.path.join(DATASET_PATH, 'centralised', 'TINYIMAGENET')
    general_directory = os.path.join(RESULTS_PATH, 'centralised', 'TINYIMAGENET', timestamp)
    results_directory = os.path.join(RESULTS_PATH, 'centralised', 'TINYIMAGENET', timestamp, 'results')
    models_directory = os.path.join(RESULTS_PATH, 'centralised', 'TINYIMAGENET', timestamp, 'models')
    os.makedirs(general_directory), os.makedirs(results_directory), os.makedirs(models_directory)
    configuration_file = os.path.join(general_directory, 'config.json')
    training_metrics_file = os.path.join(results_directory, 'training_metrics.csv')
    validation_metrics_file = os.path.join(results_directory, 'validation_metrics.csv')
    testing_metrics_file = os.path.join(results_directory, 'testing_metrics.csv')
    
    # ---------------------- Phase II: Preserve the configuration ----------------------
    training_configuration = {
        "Learning_Rate": lr,
        "Iterations": iterations,
        "Batch": batch,
        "Nesterov": nesterov,
        "momentum": momentum,
        "dampening": dampening,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "scheduler_rounds": scheduler_rounds,
        "scheduler_betas": scheduler_gamma
    }
    with open(configuration_file, "w+") as f:
        json.dump(training_configuration, f, indent=4)
    
    # ---------------------- Phase III: Load the dataset and prepare the DataLoaders ----------------------
    print(('-'*WIDTH).center(WIDTH))
    print('LOADING THE DATASET'.center(WIDTH))
    print(('-'*WIDTH).center(WIDTH))
    dataset = datasets.load_from_disk(
        dataset_path=dataset_directory
    )
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    split = test_dataset.train_test_split(test_size=0.5, train_size=0.5) # Test set is divided into validation and training set.
    
    train_set = train_dataset.with_transform(transform_func)
    test_set = split['train'].with_transform(transform_func)
    validation_set = split['test'].with_transform(transform_func)
    
    train_loader =  torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoadoer(validation_set, batch_size=batch, shuffle=True)
    
    # ---------------------- Phase IV: Define the device, model, loss function and optimizer ----------------------
    print(('-'*WIDTH).center(WIDTH))
    print('DEFINING MODEL AND OPTIMIZER'.center(WIDTH))
    print(('-'*WIDTH).center(WIDTH))
    device = torch.device(device)
    model = None
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
    if scheduler:
        scheduler_object = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_rounds, gamma=scheduler_gamma)
    else:
        scheduler_object = None
    
    # ---------------------- Phase V: csv file formating----------------------
    training_header = ['iteration', 'loss', 'gradients_norm', 'weights_norm']
    testing_header = ['iteration', 'test_loss', 'accuracy', 'f1score', 'precision', 'recall', 
                      'accuracy_per_0', 'accuracy_per_1', 'accuracy_per_2', 'accuracy_per_3',
                      'accuracy_per_4', 'accuracy_per_5', 'accuracy_per_6', 'accuracy_per_7',
                      'accuracy_per_8', 'accuracy_per_9']
    with open(training_metrics_file, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=training_header)
        writer.writeheader()
    with open(validation_metrics_file, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=testing_header)
        writer.writeheader()
    
    #  ---------------------- Phase VI: training and validation ----------------------
    for epoch in range(iterations):
        print(('-'*WIDTH).center(WIDTH))
        print(f'EPOCH {epoch}'.center(WIDTH))
        training_results = train_loop(train_loader, model, loss_fn, optimizer, device=device, scheduler=scheduler_object)
        validation_results = test_loop(validation_loader, model, loss_fn, device=device)
        results_packed = {
            'iteration': epoch,
            'loss': training_results[0],
            'gradients_norm': float(training_results[1]),
            'weights_norm': float(training_results[2])
        }
        validation_results['iteration'] = epoch
        with open(training_metrics_file, 'a+') as file:
            writer = csv.DictWriter(file, fieldnames=training_header)
            writer.writerow(results_packed)
        with open(validation_metrics_file, 'a+') as file:
            writer = csv.DictWriter(file, fieldnames=testing_header)
            writer.writerow(validation_results)
        print(f'TRAIN LOSS: {training_results[0]}'.center(WIDTH))
        print(f'VALIDATION LOSS: {validation_results['test_loss']}'.center(WIDTH))
        print(f'VALIDATION ACCURACY: {validation_results['accuracy']}'.center(WIDTH))
        if epoch % 10 == 0:
            print('SAVING THE MODEL'.center(WIDTH))
            checkpoint_name = os.path.join(models_directory, f"epoch_{epoch}_checkpoint.pt")
            torch.save(model.state_dict(), checkpoint_name)
        print(('-'*WIDTH).center(WIDTH))
    
    #  ---------------------- Phase VI: training and validation ----------------------
    final_test_results = test_loop(test_loader, model, loss_fn, device=device)
    final_test_results['iteration'] = epoch
    with open(testing_metrics_file, 'w+') as file:
        writer = csv.DictWriter(file, fieldnames=testing_header)
        writer.writeheader()
        writer.writerow(final_test_results)
    print(('-'*WIDTH).center(WIDTH))
    print(('TRAINING COMPLETED'*WIDTH).center(WIDTH))
    print(f'FINAL TEST LOSS: {final_test_results['test_loss']}'.center(WIDTH))
    print(f'FINAL TEST ACCURACY: {final_test_results['accuracy']}'.center(WIDTH))
    print(('-'*WIDTH).center(WIDTH))


if __name__ == '__main__':
    launch_centralised_tinyimagenet(
        lr=0.001,
        iterations=50,
        batch=32,
        device='cpu'
    )