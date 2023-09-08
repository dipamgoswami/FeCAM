from continuum.datasets import CIFAR100 as ICIFAR100
from continuum import ClassIncremental
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize,Resize
import timm
from torch.utils.data import DataLoader
from torch.nn import functional as F

train_ds = ICIFAR100(data_path="./data", train=True, download=True)
test_ds = ICIFAR100(data_path="./data", train=False, download=True)

scenario_train = ClassIncremental(train_ds, increment=10,initial_increment=10,transformations=[ToTensor(),Resize((224)),],class_order=np.arange(100).tolist() #[87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]
)
scenario_test = ClassIncremental(test_ds,increment=10,initial_increment=10,transformations=[ToTensor(),Resize(224)],class_order=np.arange(100).tolist() #[87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]
)

# deit_b_16 = timm.create_model("deit_small_patch16_224",pretrained=False).cuda()
# checkpoint = torch.load('weights/best_checkpoint.pth', map_location='cpu')  # for ablation experiments using prtrained weights from MORE paper 

# target = deit_b_16.state_dict()
# pretrain = checkpoint['model']
# transfer = {k: v for k, v in pretrain.items() if k in target and 'head' not in k}
# target.update(transfer)
# deit_b_16.load_state_dict(target)

vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()

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
    inv_covmat = np.linalg.pinv(cov)  # pseudo-inverse of an invertible matrix is same as its inverse
    left_term = np.matmul(dist, inv_covmat)
    mahal = np.matmul(left_term, dist.T) 
    return np.diagonal(mahal, 0)
    

class_mean_set = []
accuracy_history = []
cov_mat = []
shrink_cov_mat = []

shrink = True

for task_id, train_dataset in enumerate(scenario_train):
    train_loader = DataLoader(train_dataset, batch_size=512)
    X = []
    y = []
    num_cls = (task_id+1)*10
    for (img_batch,label,t) in train_loader:
        img_batch = img_batch.cuda()
        with torch.no_grad():
            out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
        X.append(out)
        y.append(label)
    X = np.concatenate(X)
    y = np.concatenate(y)

    for i in range(task_id * 10, (task_id+1)*10):
        image_class_mask = (y == i)
        class_mean_set.append(np.mean(X[image_class_mask],axis=0))
        cov = np.cov(X[image_class_mask].T)
        cov_mat.append(cov)
        if shrink:
            shrink_cov_mat.append(shrink_cov(cov))
    norm_cov_mat = normalize_cov(shrink_cov_mat)


    test_ds = scenario_test[:task_id+1]
    test_loader = DataLoader(test_ds, batch_size=512)
    correct , total = 0 , 0
    for (img_batch,label,t) in test_loader:
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