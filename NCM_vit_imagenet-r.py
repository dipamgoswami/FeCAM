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


class_mean_set = []
accuracy_history = []
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
        
    correct, total = 0, 0
    for (img_batch,label) in test_loader:
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
    print(f"Accuracy at {task}  {correct/total}")
    accuracy_history.append(correct/total)
print(f"incremental accuracy {np.mean(accuracy_history)}")