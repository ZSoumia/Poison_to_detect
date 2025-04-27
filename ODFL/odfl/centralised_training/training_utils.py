import torch
import datasets
import numpy as np
from torchvision import transforms
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def transform_func(
    data: datasets.arrow_dataset.Dataset
    ) -> None:
    """ Convers datasets.arrow_dataset.Dataset into a PyTorch Tensor
    Parameters
    ----------
    local_dataset: datasets.arrow_dataset.Dataset
        A local dataset that should be loaded into DataLoader
    only_test: bool [default to False]: 
        If true, only a test set will be returned
    
    Returns
    -------------
    None"""
    convert_tensor = transforms.ToTensor()
    data['image'] = [convert_tensor(img) for img in data['image']]
    return data


def train_loop(
    dataloader: torch.utils.data.DataLoader, 
    model: torch.nn, 
    loss_fn: torch.nn, 
    optimizer: torch.optim, 
    device: torch.device,
    scheduler: torch.optim.lr_scheduler = None
    ):
    losses = []
    model.train()
    for _, dic in enumerate(dataloader):
        # Assigning input + label pairs
        X = dic['image']
        y = dic['label']
        X, y = X.to(device), y.to(device)
        
        # Zero the gradients before each batch
        optimizer.zero_grad()
        
        # Making predictions
        pred = model(X)
        
        # Computing loss and its gradients
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    # Checking the norm of the gradients at the final epoch
    total_length = 0
    for p in model.parameters():
        total_length += len(p.grad.detach().flatten())
    flatten_array = torch.zeros(total_length)
    moving_index = 0
    for p in model.parameters():
        flatten_array[moving_index: (moving_index+len(p.grad.detach().flatten()))] = p.grad.detach().flatten()
        moving_index += len(p.grad.detach().flatten())
    gradients_norm = flatten_array.norm(p=1)

    # Checking the norm of the weights at the final epoch
    flatten_array = torch.zeros_like(flatten_array)
    moving_index = 0
    for p in model.parameters():
        flatten_array[moving_index: (moving_index+len(p.detach().flatten()))] = p.detach().flatten()
        moving_index += len(p.detach().flatten())
    weights_norm = flatten_array.norm(p=1)
        
    return [np.mean(losses), gradients_norm, weights_norm]


def test_loop(
    dataloader: torch.utils.data.DataLoader, 
    model: torch.nn, 
    loss_fn: torch.nn, 
    device: torch.device
    ):
    evaluation_results = {}
    model.eval()
    correct, total = 0, 0
    y_pred = []
    y_true = []
    losses = []
    
    with torch.no_grad():
        for _, dic in enumerate(dataloader):
            # Assigning input + label pairs
            X = dic['image']
            y = dic['label']
            X, y = X.to(device), y.to(device)
            
            # Making predictions
            pred = model(X)
            total += y.size(0)
            
            # Evaluating the accuracy of the prediction
            test_loss = loss_fn(pred, y).item()
            losses.append(test_loss)
            
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()            
            y_pred.append(pred)
            y_true.append(y)
    
    evaluation_results['test_loss'] = np.mean(losses)
    evaluation_results['accuracy'] = correct / total

    y_true = [item.item() for sublist in y_true for item in sublist]
    y_pred = [item.item() for sublist in y_pred for item in sublist]

    evaluation_results['f1score'] = f1_score(y_true, y_pred, average="macro")
    evaluation_results['precision'] = precision_score(y_true, y_pred, average="macro")
    evaluation_results['recall'] = recall_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    accuracy_per_class = cm.diagonal()
    accuracy_per_class_expanded = {
        f'accuracy_per_{class_id}': value for class_id, value in enumerate(accuracy_per_class)
    }
    evaluation_results.update(accuracy_per_class_expanded)
    
    return evaluation_results