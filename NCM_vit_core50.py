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


vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()

class_total_set = [0 for i in range(50)]
class_count_set = [0 for i in range(50)]
class_mean_set = [0 for i in range(50)]
accuracy_history = []


for task_id in range(8):
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

    class_mean_set = np.array(class_mean_set)
    test_ds = Core50Dataset(train=False,task_id=0,transform=Compose([Resize(256),CenterCrop(224),ToTensor()]))
    test_loader = DataLoader(test_ds, batch_size=512)
    correct , total = 0 , 0
    for (img_batch,label) in tqdm(test_loader,desc=f"Testing {task_id}",total=len(test_loader)):
        img_batch = img_batch.cuda()
        with torch.no_grad():
            out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
        predictions = []
        for single_image in out:
            distance = single_image - class_mean_set
            norm = np.linalg.norm(distance,ord=2,axis=1)
            pred = np.argmin(norm)
            predictions.append(pred)
        predictions = torch.tensor(predictions)
        correct += (predictions.cpu() == label.cpu()).sum()
        total += label.shape[0]
    print(f"Accuracy at {task_id} {correct/total}")
    accuracy_history.append(correct/total)

print(f"average incremental accuracy {round(np.mean(np.array(accuracy_history))* 100,2)} ")