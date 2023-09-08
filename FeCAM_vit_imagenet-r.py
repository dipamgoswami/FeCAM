import torch 
from torchvision.datasets import  imagenet
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import timm
from torch.nn import functional as F
import numpy as np
import os
from PIL import Image

torch.manual_seed(42)
dataset_root = "./data"
class Imagenet_r(Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images = images
        self.labels = labels


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]),self.labels[index]
    
def get_dataset(id=0):
    images = []
    labels = []
    paths = os.listdir("./data/imagenet_r")
    for i in range(id*20,(id+1)*20):
        for j in os.listdir(os.path.join("./data/imagenet_r",paths[i])):
            images.append(Image.open(os.path.join("./data/imagenet_r",paths[i],j)).convert("RGB"))
            labels.append(i)
    return Imagenet_r(images,labels,transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()]))


def shrink_cov(cov):
    diag_mean = np.mean(np.diagonal(cov))
    off_diag = np.copy(cov)
    np.fill_diagonal(off_diag,0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = np.eye(cov.shape[0])
    alpha1 = 10
    alpha2  = 10
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


class_mean_set = []
accuracy_history = []
cov_mat = []
shrink_cov_mat = []

shrink = True

for task in range(10):
    print(f"Task {task}")
    imagenet_r = get_dataset(task)
    size_of_dataset = len(imagenet_r)
    train_dataset , test_dataset_ = torch.utils.data.random_split(imagenet_r, [int(0.8 * size_of_dataset), size_of_dataset - int(0.8 * size_of_dataset)])

    if task > 0:
        test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_])
    else:
        test_dataset = test_dataset_

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

    vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()
    
    X,y = [],[]
    num_cls = (task+1)*20
    for (img_batch,label) in train_loader:
        img_batch = img_batch.cuda()
        with torch.no_grad():
            out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
        X.append(out)
        y.append(label)
    X = np.concatenate(X)
    y = np.concatenate(y)
    for i in range(20 * task,20 * (task+1)):
        image_class_mask = (y == i)
        class_mean_set.append(np.mean(X[image_class_mask],axis=0))
        cov = np.cov(X[image_class_mask].T)
        cov_mat.append(cov)
        if shrink:
            shrink_cov_mat.append(shrink_cov(cov))
    norm_cov_mat = normalize_cov(shrink_cov_mat)

    correct, total = 0, 0
    for (img_batch,label) in test_loader:
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

    print(f"Accuracy at {task}  {correct/total}")
    accuracy_history.append(correct/total)

print(f"incremental accuracy {np.mean(accuracy_history)}")