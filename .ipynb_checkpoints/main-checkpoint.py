from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import numpy as np
import torch
import time
import argparse
import os
import sys


#local imports
from datasets.cifar10 import CIFAR10
from datasets.mnist import MNIST
from models.svm import OneVsRestSVM


# command line arguments
parser = argparse.ArgumentParser(description='Multi-Class SVM Image Classification')
parser.add_argument('--name', type=str, help='Experiment Name')
parser.add_argument('--root', type=str, help='Dataset Directory')
parser.add_argument('--dataset', type=str, help='Dataset')
parser.add_argument('--samples', type=int, default=30000, help='num samples to use')
parser.add_argument('--scale', action='store_true', default=False, help='Use Scaling')
parser.add_argument('--full', action='store_true', default=False, help='Use entire dataset')
parser.add_argument('--pca', action='store_true', default=False, help='Use PCA')
parser.add_argument('--kernel', type=str, help='kernel to use')
parser.add_argument('--C', nargs='+', type=float, default=[1.0], help='C values')
parser.add_argument('--gamma', nargs='+', type=float, default=[1.0], help='gamma values')
parser.add_argument('--seed', type=int, default=42, help='Random Seed')
args = parser.parse_args()

dataset = args.dataset
assert dataset in ['cifar10', 'mnist']
kernel = args.kernel
assert kernel in ['linear', 'rbf']



experiment_dir = f'models/saves/{dataset}/{args.name}'
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
    print(f"Save Directory '{experiment_dir}' created successfully.")
else:
    print(f"Directory '{experiment_dir}' already exists. Exiting program.")
    sys.exit()

n_samples, C_vals, gamma_vals, scale, full, pca = \
args.samples, args.C, args.gamma, args.scale, args.full, args.pca

with open(f'{experiment_dir}/results.txt', 'a') as file:
    if kernel == 'linear':
        file.write(f'args:{kernel} kernel, C={C_vals}, scale={scale}, pca={pca}, samples={n_samples}\n\n')
    else:
         file.write(f'args:{kernel} kernel, C={C_vals}, gamma={gamma_vals}, scale={scale}, pca={pca}, samples={n_samples}\n\n')

root = os.path.join(args.root, dataset)

# loading data
if dataset == 'cifar10':
    train_ds = CIFAR10(root, train=True)
elif dataset == 'mnist':
    train_ds = MNIST(root, train=True)
    
X_train = train_ds.get_images()
y_train = train_ds.get_labels()

# preprocessing
if not full:
    X_train = X_train[:n_samples]
    y_train = y_train[:n_samples]

if scale:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

if pca:
    pca_solver = PCA(n_components=0.90) # keep 95% of the variance
    pca_solver.fit(X_train)
    X_train = pca.transform(X_train)
    
sub_experiment_num = 0
for C in C_vals:
    for gamma in gamma_vals:
            
        sub_experiment_num += 1
        sub_dir = f'{experiment_dir}/{sub_experiment_num}'
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        else:
            print(f"Duplicate directory '{sub_dir}' already exists. Exiting program.")
            sys.exit()


        with open(f'{sub_dir}/params.txt', 'a') as file:
            file.write('Params: \n')
            file.write(f'C = {C}\n')
            if kernel == 'rbf':
                  file.write(f'gamma = {gamma}\n')

            
        # training
        n_classes = 10
        model = OneVsRestSVM(n_classes, sub_dir)
        model.fit(X_train, y_train, kernel=kernel, dataset='cifar', C=C, gamma=gamma)

        # testing
        if dataset == 'cifar10':
            train_ds = CIFAR10(root, train=False)
        elif dataset == 'mnist':
            train_ds = MNIST(root, train=False)

        X_test = train_ds.get_images()
        y_test = train_ds.get_labels()

        if scale:
            X_test = scaler.transform(X_test)

        if pca:
            X_test = pca_solver.transform(X_test)


        accuracy = model.test(X_test, y_test)
        with open(f'{experiment_dir}/results.txt', 'a') as file:
            if kernel == 'linear':
                file.write(f'Test Accuracy for {kernel} kernel, C={C}: {accuracy * 100:.2f}%\n')
            else:
                 file.write(f'Test Accuracy for {kernel} kernel, C={C}, gamma={gamma}: {accuracy * 100:.2f}%\n')
                    
        if kernel == 'linear':
            print(f"Test Accuracy for {kernel} kernel, C={C}: {accuracy * 100:.2f}%")    
        else: 
            print(f"Test Accuracy for {kernel} kernel, C={C}, gamma={gamma}: {accuracy * 100:.2f}%")