from continuum.datasets import Core50
from continuum import ContinualScenario
from tqdm import tqdm
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize,Resize,CenterCrop
import timm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import Dataset
import os
from PIL import Image


class Core50Dataset(Dataset):
    def __init__(self, train=True,task_id=0, transform=None):
        self.data = []
        self.labels = []
        self.root = "./data/core50_128x128"
        self.transform = transform
        if train:
            train_set = ["s1","s2","s4","s5","s6","s8","s9","s11"]
            for object_folder in os.listdir(f"{self.root}/{train_set[task_id]}"):
                for image in os.listdir(f"{self.root}/{train_set[task_id]}/{object_folder}"):
                    self.data.append(f"{self.root}/{train_set[task_id]}/{object_folder}/{image}")
                    self.labels.append(int(object_folder[1:])-1)
        else:
            test_set = ["s3","s7","s10"]
            for session_folder in test_set:
                for object_folder in os.listdir(f"{self.root}/{session_folder}"):
                    for image in os.listdir(f"{self.root}/{session_folder}/{object_folder}"):
                        self.data.append(f"{self.root}/{session_folder}/{object_folder}/{image}")
                        self.labels.append(int(object_folder[1:])-1)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.data[idx]))
        return image, self.labels[idx]
    

def shrink_cov(cov):
    diag_mean = np.mean(np.diagonal(cov))
    off_diag = np.copy(cov)
    np.fill_diagonal(off_diag,0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = np.eye(cov.shape[0])
    alpha1 = 1
    alpha2  = 0
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_

def normalize_cov(cov_mat):
    norm_cov_mat = []
    for cov in cov_mat:
        sd = np.sqrt(np.diagonal(cov))  # standard deviations of the variables
        cov = cov/(np.matmul(np.expand_dims(sd,1),np.expand_dims(sd,0)))
        norm_cov_mat.append(cov)

    return norm_cov_mat


def _mahalanobis(dist, cov=None):
    if cov is None:
        cov = np.eye(768)
    inv_covmat = np.linalg.pinv(cov)
    left_term = np.matmul(dist, inv_covmat)
    mahal = np.matmul(left_term, dist.T) 
    return np.diagonal(mahal, 0)


vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()

class_total_set = [0 for i in range(50)]
class_count_set = [0 for i in range(50)]
class_mean_set = [0 for i in range(50)]
cov_mat = []
shrink_cov_mat = []

shrink = True
accuracy_history = []


for task_id in range(8):
    num_cls = 50
    print(f"Task {task_id}")
    train_dataset = Core50Dataset(train=True,task_id=task_id,transform=Compose([Resize(256),CenterCrop(224),ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=1024)
    X = []
    y = []
    print("Starting training")
    for (img_batch,label) in tqdm(train_loader,desc=f"Training {task_id}",total=len(train_loader)):
        img_batch = img_batch.cuda()
        with torch.no_grad():
            out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
        X.append(out)
        y.append(label)
    X = np.concatenate(X)
    y = np.concatenate(y)
    for i in range(0, 50):
        image_class_mask = (y == i)
        class_total = np.sum(X[image_class_mask],axis=0)
        class_count = np.sum(image_class_mask)
        class_total_set[i] += class_total
        class_count_set[i] += class_count
        class_mean_set[i] = class_total_set[i]/class_count_set[i]

        cov = np.cov(X[image_class_mask].T)
        if task_id == 0:
            cov_mat.append(cov)
            if shrink:
                shrink_cov_mat.append(shrink_cov(cov))
        else:
            if shrink:
                cov = shrink_cov(cov)
            shrink_cov_mat[i] = (shrink_cov_mat[i]+cov)/2  # Average of cov matrices from previous and current domain
    norm_cov_mat = normalize_cov(shrink_cov_mat)

    class_mean_set = np.array(class_mean_set)

    test_ds = Core50Dataset(train=False,task_id=0,transform=Compose([Resize(256),CenterCrop(224),ToTensor()]))
    test_loader = DataLoader(test_ds, batch_size=512)
    correct , total = 0 , 0
    for (img_batch,label) in tqdm(test_loader,desc=f"Testing {task_id}",total=len(test_loader)):
        img_batch = img_batch.cuda()
        with torch.no_grad():
            out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
        predictions = []

        maha_dist = []
        for cl in range(num_cls):
            distance = out - class_mean_set[cl]
            dist = _mahalanobis(distance, norm_cov_mat[cl])
            maha_dist.append(dist)
        maha_dist = np.array(maha_dist)
        pred = np.argmin(maha_dist.T, axis=1)
        predictions.append(pred)

        predictions = torch.tensor(np.array(predictions))
        correct += (predictions.cpu() == label.cpu()).sum()
        total += label.shape[0]
        
    print(f"Accuracy at {task_id} {correct/total}")
    accuracy_history.append(correct/total)

print(f"average incremental accuracy {round(np.mean(np.array(accuracy_history))* 100,2)} ")