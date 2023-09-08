import torch
import os
import pickle
import numpy as np
import torch.nn as nn
from torch import linalg as LA
from torch.utils.data import DataLoader


def compute_base_common_cov(train_loader, model):
    if model.args["dataset"] == 'cifar100' and model.args["tukey"] and os.path.exists('cifar_common_cov_tukey.pkl'):
        with open('cifar_common_cov_tukey.pkl', 'rb') as f:
            common_cov = pickle.load(f)
    elif model.args["dataset"] == 'cifar100' and not model.args["tukey"] and os.path.exists('cifar_common_cov.pkl'):
        with open('cifar_common_cov.pkl', 'rb') as f:
            common_cov = pickle.load(f)
    elif model.args["dataset"] == 'imagenet100' and model.args["tukey"] and os.path.exists('imagenet100_common_cov_tukey.pkl'):
        with open('imagenet100_common_cov_tukey.pkl', 'rb') as f:
            common_cov = pickle.load(f)
    elif model.args["dataset"] == 'imagenet100' and not model.args["tukey"] and os.path.exists('imagenet100_common_cov.pkl'):
        with open('imagenet100_common_cov.pkl', 'rb') as f:
            common_cov = pickle.load(f)
    elif model.args["dataset"] == 'cub200' and model.args["tukey"] and os.path.exists('cub200_common_cov_tukey.pkl'):
        with open('cub200_common_cov_tukey.pkl', 'rb') as f:
            common_cov = pickle.load(f)
    elif model.args["dataset"] == 'cub200' and not model.args["tukey"] and os.path.exists('cub200_common_cov.pkl'):
        with open('cub200_common_cov.pkl', 'rb') as f:
            common_cov = pickle.load(f)
    else:
        common_cov = compute_common_cov(train_loader, model)

        if model.args["dataset"] == 'cifar100':
            if model.args["tukey"]:
                with open('cifar_common_cov_tukey.pkl', 'wb') as f:
                    pickle.dump(common_cov, f)
            else:
                with open('cifar_common_cov.pkl', 'wb') as f:
                    pickle.dump(common_cov, f)
        elif model.args["dataset"] == 'imagenet100':
            if model.args["tukey"]:
                with open('imagenet100_common_cov_tukey.pkl', 'wb') as f:
                    pickle.dump(common_cov, f)
            else:
                with open('imagenet100_common_cov.pkl', 'wb') as f:
                    pickle.dump(common_cov, f)
        elif model.args["dataset"] == 'cub200':
            if model.args["tukey"]:
                with open('cub200_common_cov_tukey.pkl', 'wb') as f:
                    pickle.dump(common_cov, f)
            else:
                with open('cub200_common_cov.pkl', 'wb') as f:
                    pickle.dump(common_cov, f)
    
    return common_cov

def compute_common_cov(train_loader, model):
    if model.args["dataset"] == 'imagenet100':
        cov = model._extract_vectors_common_cov(train_loader)
    else:
        vectors, _ = model._extract_vectors(train_loader)
        if model.args["tukey"]:
            vectors = model._tukeys_transform(vectors)
        cov = torch.tensor(np.cov(vectors.T))
    return cov

def compute_new_common_cov(train_loader, model):
    cov = compute_common_cov(train_loader, model)
    if model.args["shrink"]:
        cov = model.shrink_cov(cov)
    ratio = (model._known_classes/model._total_classes)

    common_cov = ratio*model._common_cov + (1-ratio)*cov
    return common_cov

def compute_base_cov(model):
    if model.args["dataset"] == 'cifar100' and model.args["tukey"] and os.path.exists('cifar_cov_tukey.pkl'):
        with open('cifar_cov_tukey.pkl', 'rb') as f:
            model._cov_mat = pickle.load(f)
    elif model.args["dataset"] == 'cifar100' and not model.args["tukey"] and os.path.exists('cifar_cov.pkl'):
        with open('cifar_cov.pkl', 'rb') as f:
            model._cov_mat = pickle.load(f)
    elif model.args["dataset"] == 'imagenet100' and model.args["tukey"] and os.path.exists('imagenet100_cov_tukey.pkl'):
        with open('imagenet100_cov_tukey.pkl', 'rb') as f:
            model._cov_mat = pickle.load(f)
    elif model.args["dataset"] == 'imagenet100' and not model.args["tukey"] and os.path.exists('imagenet100_cov.pkl'):
        with open('imagenet100_cov.pkl', 'rb') as f:
            model._cov_mat = pickle.load(f)
    elif model.args["dataset"] == 'cub200' and model.args["tukey"] and os.path.exists('cub200_cov_tukey.pkl'):
        with open('cub200_cov_tukey.pkl', 'rb') as f:
            model._cov_mat = pickle.load(f)
    elif model.args["dataset"] == 'cub200' and not model.args["tukey"] and os.path.exists('cub200_cov.pkl'):
        with open('cub200_cov.pkl', 'rb') as f:
            model._cov_mat = pickle.load(f)
    else:
        compute_new_cov(model)

        if model.args["dataset"] == 'cifar100':
            if model.args["tukey"]:
                with open('cifar_cov_tukey.pkl', 'wb') as f:
                    pickle.dump(model._cov_mat, f)
            else:
                with open('cifar_cov.pkl', 'wb') as f:
                    pickle.dump(model._cov_mat, f)
        elif model.args["dataset"] == 'imagenet100':
            if model.args["tukey"]:
                with open('imagenet100_cov_tukey.pkl', 'wb') as f:
                    pickle.dump(model._cov_mat, f)
            else:
                with open('imagenet100_cov.pkl', 'wb') as f:
                    pickle.dump(model._cov_mat, f)
        elif model.args["dataset"] == 'cub200':
            if model.args["tukey"]:
                with open('cub200_cov_tukey.pkl', 'wb') as f:
                    pickle.dump(model._cov_mat, f)
            else:
                with open('cub200_cov.pkl', 'wb') as f:
                    pickle.dump(model._cov_mat, f)

def compute_new_cov(model):
    for class_idx in range(model._known_classes, model._total_classes):
        data, targets, idx_dataset = model.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                mode='test', ret_data=True)
        idx_loader = DataLoader(idx_dataset, batch_size=model.args["batch_size"], shuffle=False, num_workers=4)
        vectors, _ = model._extract_vectors(idx_loader)
        if model.args["tukey"]:
            vectors = model._tukeys_transform(vectors)
        
        cov = torch.tensor(np.cov(vectors.T))
        if model.args["shrink"]:
            cov = model.shrink_cov(cov)

        model._cov_mat.append(cov)
