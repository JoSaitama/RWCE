import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch.nn.functional import softmax
from collections import defaultdict
from sklearn.metrics import top_k_accuracy_score
import torch.nn.functional as F

import os
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_softmax_and_ranks(val_loader, model, args):

    model.eval()
    logits_all = []  # Initialize list for logits
    logit_ranks_all = []  # Initialize list for logit ranks
    print(f'Computing the true label rank in softmax scores).')

    with torch.no_grad():  # Turn off gradients, as we are in test mode
        for x, targets in tqdm(val_loader):  

            x = x.to(args.gpu, non_blocking=True)  # Move inputs to GPU
            targets = targets.to(args.gpu, non_blocking=True)  # Move labels to GPU

            # Forward pass
            outputs = model(x)  # This gets the logits
            softmax_scores = F.softmax(outputs, dim=1).cpu().numpy()

            # Get the ranks
            _, indices = torch.sort(outputs, descending=True)
            ranks = torch.zeros_like(indices)
            for i in range(outputs.shape[0]):
                ranks[i][indices[i]] = torch.arange(outputs.shape[1], device='cuda')

            logit_ranks = ranks[torch.arange(outputs.shape[0]), targets].detach().cpu().numpy()
            logits = softmax_scores

            logits_all.append(logits)
            logit_ranks_all.append(logit_ranks)

    logits_all = np.concatenate(logits_all, axis=0)
    logit_ranks_all = np.concatenate(logit_ranks_all, axis=0)

    return logits_all, logit_ranks_all

class DataWithRanks(torch.utils.data.Dataset):
    def __init__(self, dataset, softmax_scores, ranks):
        self.dataset = dataset
        self.softmax_scores = softmax_scores
        self.ranks = ranks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        softmax_scores = self.softmax_scores[idx]
        rank = self.ranks[idx]
        #print(f'Creating dataset with softmax_scores and true label rank.')
        return {'image': image, 'label': label, 'softmax': softmax_scores, 'rank': rank}


def calc_top_k_accuracy_per_class(val_loader, model, args, k):
    model.eval()

    class_targets = defaultdict(list)
    class_scores = defaultdict(list)

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            softmax_scores = F.softmax(output, dim=1).cpu().numpy()

            # Divide the targets and scores into classes
            for score, t in zip(softmax_scores, target.cpu().numpy()):
                class_targets[t].append(t)
                class_scores[t].append(score)

    # Compute top-k accuracy for each class
    top_k_acc_per_class = {}
    for class_label in class_targets.keys():
        targets = class_targets[class_label]
        scores = class_scores[class_label]

        # Compute top-k accuracy manually
        correct = 0
        for target, score in zip(targets, scores):
            if target in np.argsort(score)[-k:]:
                correct += 1

        top_k_acc = correct / len(targets)
        top_k_acc_per_class[class_label] = top_k_acc

    return top_k_acc_per_class

def accuracy_matrix(val_loader, model, args, num_class):
    matrix = []
    for k in range(1, num_class+1):
        cls_test_2 = calc_top_k_accuracy_per_class(val_loader, model, args, k)
    
        cls_test_3 = [cls_test_2[i] for i in range(len(cls_test_2))]

        matrix.append(cls_test_3)

    return matrix

def calc_top_k_accuracy_per_class2(softmax_scores, targets, num_class, k):
    class_targets = defaultdict(list)
    class_scores = defaultdict(list)

    # Divide the targets and scores into classes
    for score, t in zip(softmax_scores, targets):
        class_targets[t].append(t)
        class_scores[t].append(score)

    # Compute top-k accuracy for each class
    top_k_acc_per_class = {}
    for class_label in class_targets.keys():
        targets = class_targets[class_label]
        scores = class_scores[class_label]

        # Compute top-k accuracy manually
        correct = 0
        for target, score in zip(targets, scores):
            if target in np.argsort(score)[-k:]:
                correct += 1

        top_k_acc = correct / len(targets)
        top_k_acc_per_class[class_label] = top_k_acc

    return top_k_acc_per_class

def load_or_compute_acc_matrix(softmax_scores, targets, num_class, checkpoint_path):
    if os.path.exists(checkpoint_path):
        acc_matrix = np.load(checkpoint_path, allow_pickle=True)
        
        # Check if the computation was fully completed
        if len(acc_matrix) == num_class:
            print("Full acc_matrix found, loading without recomputation.")
            return acc_matrix
        else:
            print(f"Partial acc_matrix found with {len(acc_matrix)} entries. Resuming computation...")
    else:
        print("No acc_matrix found. Starting computation...")
        acc_matrix = []

    # Continue computation from where it left off
    acc_matrix = accuracy_matrix2(softmax_scores, targets, num_class, checkpoint_path, acc_matrix=acc_matrix)
    
    return acc_matrix

def accuracy_matrix2(softmax_scores, targets, num_class, checkpoint_path, checkpoint_interval=10, acc_matrix=None):
    if acc_matrix is None:
        acc_matrix = []
    
    for k in range(len(acc_matrix) + 1, num_class + 1):
        cls_test_2 = calc_top_k_accuracy_per_class2(softmax_scores, targets, num_class, k)
        cls_test_3 = [cls_test_2[i] for i in range(len(cls_test_2))]
        acc_matrix.append(cls_test_3)
        
        if k % checkpoint_interval == 0 or k == num_class:
            np.save(checkpoint_path, np.array(acc_matrix, dtype=object))
            print(f"Checkpoint saved at k={k}")

    return acc_matrix

def calc_top_k_accuracy_whole_dataset(softmax_scores, targets, k):
    correct = 0
    total = len(targets)
    
    for score, target in zip(softmax_scores, targets):
        top_k_preds = np.argsort(score)[-k:]
        if target in top_k_preds:
            correct += 1

    top_k_accuracy = correct / total
    return top_k_accuracy

# def top_acc_vector(softmax_scores, targets, num_class):
#     acc_list = []

#     for k in range(1, num_class+1):
#         top_k_accuracy = calc_top_k_accuracy_whole_dataset(softmax_scores, targets, k)
#         acc_list.append(top_k_accuracy)

#     return acc_list

def load_or_compute_acc_vector(softmax_scores, targets, num_class, checkpoint_path):
    if os.path.exists(checkpoint_path):
        acc_vector = np.load(checkpoint_path)
        
        # Check if the computation was fully completed
        if len(acc_vector) == num_class:
            print("Full acc_vector found, loading without recomputation.")
            return list(acc_vector)
        else:
            print(f"Partial acc_vector found with {len(acc_vector)} entries. Resuming computation...")
    else:
        print("No acc_vector found. Starting computation...")
        acc_vector = []

    # Continue computation from where it left off
    acc_vector = top_acc_vector(softmax_scores, targets, num_class, checkpoint_path, acc_vector=acc_vector)
    
    return acc_vector

def top_acc_vector(softmax_scores, targets, num_class, checkpoint_path, checkpoint_interval=10, acc_vector=None):
    if acc_vector is None:
        acc_vector = []
    
    for k in range(len(acc_vector) + 1, num_class + 1):
        top_k_accuracy = calc_top_k_accuracy_whole_dataset(softmax_scores, targets, k)
        acc_vector.append(top_k_accuracy)
        
        if k % checkpoint_interval == 0 or k == num_class:
            np.save(checkpoint_path, np.array(acc_vector))
            print(f"Checkpoint saved at k={k}")

    return acc_vector

def load_or_compute_tensor_acc_vector(softmax_scores, targets, num_class, checkpoint_path):
    if os.path.exists(checkpoint_path):
        acc_vector = torch.load(checkpoint_path)
        
        # Check if the computation was fully completed
        if len(acc_vector) == num_class:
            print("Full acc_vector found, loading without recomputation.")
            return list(acc_vector)
        else:
            print(f"Partial acc_vector found with {len(acc_vector)} entries. Resuming computation...")
    else:
        print("No acc_vector found. Starting computation...")
        acc_vector = []

    # Continue computation from where it left off
    acc_vector = top_acc_vector(softmax_scores, targets, num_class, checkpoint_path, acc_vector=acc_vector)
    
    return acc_vector

def top_tensor_acc_vector(softmax_scores, targets, num_class, checkpoint_path, checkpoint_interval=10, acc_vector=None):
    if acc_vector is None:
        acc_vector = []
    
    for k in range(len(acc_vector) + 1, num_class + 1):
        top_k_accuracy = calc_top_k_accuracy_whole_dataset(softmax_scores, targets, k)
        acc_vector.append(top_k_accuracy)
        
        if k % checkpoint_interval == 0 or k == num_class:
            torch.save(torch.tensor(acc_vector), checkpoint_path)
            print(f"Checkpoint saved at k={k}")

    return acc_vector

def load_dataset(args, dataset, data_folder='CCC_model'):
    '''
    Load softmax scores and labels for a dataset
    
    Input:
        - dataset: string specifying dataset. Options are 'imagenet', 'cifar-100', 'places365', 'inaturalist'
        - data_folder: string specifying folder containing the <dataset name>.npz files

    Output: softmax_scores, labels
        
    '''
    # assert dataset in ['imagenet', 'cifar100', 'places365', 'inaturalist']
    
    
    data = np.load(f'{data_folder}/{dataset}_{args.epochs}_{args.lr}.npz')
    # data = np.load(f'{data_folder}/{dataset}.npz')
    softmax_scores = data['softmax']
    labels = data['labels']
    
    return softmax_scores, labels

def compute_true_label_rank(softmax_scores, labels):

    ranks = np.zeros(softmax_scores.shape[0], dtype=int)

    for i, (scores, label) in enumerate(zip(softmax_scores, labels)):
        # Sort scores in descending order and get the sorted indices
        sorted_indices = np.argsort(scores)[::-1]
        # Find the index of the true label in the sorted indices
        rank = np.where(sorted_indices == label)[0][0] + 1
        ranks[i] = rank

    return ranks