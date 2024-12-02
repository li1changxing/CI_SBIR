
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def map_sake(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    mean_mAP = []
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        mean_mAP.append(mapi)
    return mean_mAP

def prec_sake(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    # compute precision for two modalities
    mean_prec = []
    for fi in range(predicted_features_query.shape[0]):
        prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        mean_prec.append(prec)
    # print("precision for all samples: ", np.nanmean(mean_prec))
    return np.nanmean(mean_prec)

def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]  # total retrieved samples
    tot_pos = np.sum(pos_flag)  # total true position

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]  # sorted true positive
    fp = np.logical_not(tp)  # sorted false positive

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]  # select top-k true position
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap

def VOCap(rec, prec):
    mrec = np.append(0, rec)  # put 0 in the first element
    mrec = np.append(mrec, 1)  # put 1 in the last element

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):  # sort mpre, the smaller, the latter
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap

def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top

def valid(photo_loader, sketch_loader, model,k,all=False):
    gallery_reprs = []
    gallery_reprs_skt = []
    gallery_labels = []
    gallery_labels_skt = []
    model.eval()
    with torch.no_grad():
        for idx,(photo, label) in enumerate(tqdm(photo_loader)):
            photo, label = photo.cuda(), label
            photo_reprs = model(photo,'im').cpu()
            gallery_reprs.append(photo_reprs)
            for i in label:
                gallery_labels.append(i)

        gallery_reprs = F.normalize(torch.cat(gallery_reprs))
        gallery_labels = torch.tensor(gallery_labels)

        for idx,(sketch, label) in enumerate(tqdm(sketch_loader)):
            sketch, label = sketch.cuda(), label
            sketch_reprs = F.normalize(model(sketch,'sk')).cpu()
            gallery_reprs_skt.append(sketch_reprs)
            for i in label:
                gallery_labels_skt.append(i)

        gallery_reprs_skt = F.normalize(torch.cat(gallery_reprs_skt))
        gallery_labels_skt = torch.tensor(gallery_labels_skt)

    test_features_img = nn.functional.normalize(gallery_reprs, dim=1, p=2)
    test_features_skt = nn.functional.normalize(gallery_reprs_skt, dim=1, p=2)
    ############################################################################
    # Step 2: similarity
    sim = torch.mm(test_features_skt, test_features_img.T)
    k = {'map': test_features_img.shape[0], 'precision': k}
    ############################################################################
    # Step 3: evaluate
    aps = map_sake(test_features_img.numpy(), gallery_labels.numpy(),
                   test_features_skt.numpy(), gallery_labels_skt.numpy(),sim, k=k['map'])
    prec = prec_sake(test_features_img.numpy(), gallery_labels.numpy(),
                   test_features_skt.numpy(), gallery_labels_skt.numpy(),sim,k=k['precision'])
    print('map{}: {:.4f} prec{}: {:.4f}'.format(k['map'], np.mean(aps), k['precision'], prec))
    if all:
        return np.mean(aps), prec

    return np.mean(aps)

