import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import torch.nn.functional as F

EPSILON = 1e-8


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self.topk = 5

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if self.args["full_cov"] or self.args["diagonal"]:
            y_pred, y_true = self._eval_maha(self.test_loader, self._init_protos, self._protos)
            maha_accy = self._evaluate(y_pred, y_true)
        else:
            maha_accy = None

        nme_accy = None

        return cnn_accy, nme_accy, maha_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    
    def _eval_maha(self, loader, init_means, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = self._maha_dist(vectors, init_means, class_means)
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]
    

    def _maha_dist(self, vectors, init_means, class_means):
        vectors = torch.tensor(vectors).to(self._device)
        if self.args["tukey"] and self._cur_task > 0:
            vectors = self._tukeys_transform(vectors)
        maha_dist = []
        for class_index in range(self._total_classes):
            if self._cur_task == 0:
                dist = self._mahalanobis(vectors, init_means[class_index])
            else:
                if self.args["ncm"]:
                    dist = self._mahalanobis(vectors, class_means[class_index])
                elif self.args["full_cov"]:
                    if self.args["per_class"]:
                        if self.args["norm_cov"]:
                            dist = self._mahalanobis(vectors, class_means[class_index], self._norm_cov_mat[class_index])
                        elif self.args["shrink"]:
                            dist = self._mahalanobis(vectors, class_means[class_index], self._cov_mat_shrink[class_index])
                        else:
                            dist = self._mahalanobis(vectors, class_means[class_index], self._cov_mat[class_index])
                    else:
                        dist = self._mahalanobis(vectors, class_means[class_index], self._common_cov)
                elif self.args["diagonal"]:
                    if self.args["per_class"]:
                        dist = self._mahalanobis(vectors, class_means[class_index], self._diag_mat[class_index])
            maha_dist.append(dist)
        maha_dist = np.array(maha_dist)  # [nb_classes, N]  
        return maha_dist

    def _mahalanobis(self, vectors, class_means, cov=None):
        if self.args["tukey"] and self._cur_task > 0:
            class_means = self._tukeys_transform(class_means)
        x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
        if cov is None:
            cov = torch.eye(self._network.feature_dim)  # identity covariance matrix for euclidean distance
        inv_covmat = torch.linalg.pinv(cov).float().to(self._device)
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0).cpu().numpy()
    
    def diagonalization(self, cov):
        diag = cov.clone()
        cov_ = cov.clone()
        cov_.fill_diagonal_(0.0)
        diag = diag - cov_
        return diag
    
    def shrink_cov(self, cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag*mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0])
        alpha1 = self.args["alpha1"]
        alpha2  = self.args["alpha2"]
        cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
        return cov_
    
    def normalize_cov(self):
        if self.args["shrink"]:
            cov_mat = self._cov_mat_shrink
        else:
            cov_mat = self._cov_mat
        norm_cov_mat = []
        for cov in cov_mat:
            sd = torch.sqrt(torch.diagonal(cov))  # standard deviations of the variables
            cov = cov/(torch.matmul(sd.unsqueeze(1),sd.unsqueeze(0)))
            norm_cov_mat.append(cov)
            
        print(len(norm_cov_mat))
        return norm_cov_mat    
    
    def normalize_cov2(self, cov):
        diag = torch.diagonal(cov)
        norm = torch.linalg.norm(diag)
        cov = cov /norm
        return cov

    def _tukeys_transform(self, x):
        beta = self.args["beta"]
        x = torch.tensor(x)
        if beta == 0:
            return torch.log(x)
        else:
            return torch.pow(x, beta)

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def _extract_vectors_common_cov(self, loader):
        self._network.eval()
        vectors, covs = [], []
        for i, (_, _inputs, _) in enumerate(loader):
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            if i % 20 == 0:
                vecs = np.concatenate(vectors)
                if self.args["tukey"]:
                    vecs = self._tukeys_transform(vecs)
                covs.append(np.cov(vecs.T))
                vectors = []
        
        if len(vectors) > 4:
            vecs = np.concatenate(vectors)
            if self.args["tukey"]:
                vecs = self._tukeys_transform(vecs)
            covs.append(np.cov(vecs.T))

        cov = np.mean(covs, axis=0)
        return torch.tensor(cov)
