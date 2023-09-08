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

scenario_train = ClassIncremental(train_ds, increment=10,initial_increment=10,transformations=[ToTensor(),Resize((224)),],class_order=np.arange(100).tolist()) #[87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]

scenario_test = ClassIncremental(test_ds,increment=10,initial_increment=10,transformations=[ToTensor(),Resize(224)],class_order=np.arange(100).tolist()) #[87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]


vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()

class_mean_set = []
accuracy_history = []
for task_id, train_dataset in enumerate(scenario_train):
    train_loader = DataLoader(train_dataset, batch_size=512)
    X = []
    y = []
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
    test_ds = scenario_test[:task_id+1]
    test_loader = DataLoader(test_ds, batch_size=512)
    correct , total = 0 , 0
    for (img_batch,label,t) in test_loader:
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