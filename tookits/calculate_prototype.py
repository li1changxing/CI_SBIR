import torch
import numpy as np
import torch.nn as nn
def displacement(Y1, Y2, embedding_old, sigma=4.0):
    Y1 = Y1.detach().cpu().numpy()
    Y2 = Y2.detach().cpu().numpy()
    embedding_old = embedding_old.cpu().numpy()
    DY = Y2 - Y1
    distance = np.sum(
        (np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2
        , axis=2)
    W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
    W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
    displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
    return torch.tensor(displacement).cuda()
def align_loss(x, y, alpha=2):
    '''
    https://github.com/SsnL/align_uniform/blob/master/align_uniform/__init__.py
    :param x:
    :param y:
    :param alpha:
    :return:
    '''
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()
class Prototype(nn.Module):
    def __init__(self,img,skt, momentum=0.9,dim =384):
        super(Prototype, self).__init__()
        self.momentum = momentum
        self.register_buffer("center_skt", skt)
        self.register_buffer("center_img", img)
        self.fea_dim = dim

    def forward(self, x, l, modality='img'):
        class_in_batch = self.update_center(x, l, modality)

        return align_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch])

    def update_center(self, x, l, modality):
        self.center_img = self.center_img.detach()
        self.center_skt = self.center_skt.detach()

        all_l=l
        classes_in_batch, sam2cls_idx, cl_sam_counts = torch.unique(all_l, return_counts=True, sorted=True, return_inverse=True)
        center_tmp = torch.zeros(len(classes_in_batch), self.fea_dim).cuda()
        for i, idx in enumerate(sam2cls_idx):
            center_tmp[idx] += x[i]
        center_tmp = center_tmp / cl_sam_counts.unsqueeze(1)

        if modality == 'img':
            self.center_img[classes_in_batch] = self.center_img[classes_in_batch] * self.momentum + center_tmp * (1 - self.momentum)
            self.center_img[classes_in_batch] /= self.center_img[classes_in_batch].norm(p=2, dim=1, keepdim=True)
        else:
            self.center_skt[classes_in_batch] = self.center_skt[classes_in_batch] * self.momentum + center_tmp * (1 - self.momentum)
            self.center_skt[classes_in_batch] /= self.center_skt[classes_in_batch].norm(p=2, dim=1, keepdim=True)

        return classes_in_batch

    def update_shift(self,img_old,img_new,skt_old,skt_new,n,l):
        self.center_img = self.center_img.detach()
        self.center_skt = self.center_skt.detach()

        gap_img = displacement(img_old, img_new, self.center_img[n:], 4.0)
        gap_skt = displacement(skt_old, skt_new, self.center_skt[n:], 4.0)

        self.center_img[n:] = self.center_img[n:] + (1 - self.momentum)/l * gap_img
        self.center_skt[n:] = self.center_skt[n:] + (1 - self.momentum)/l * gap_skt

        # self.center_img[n:] /= self.center_img[n:].norm(p=2, dim=1, keepdim=True)
        # self.center_skt[n:] /= self.center_skt[n:].norm(p=2, dim=1, keepdim=True)

        return align_loss(self.center_img[n:], self.center_skt[n:])
