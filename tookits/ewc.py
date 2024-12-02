import torch
from tqdm import tqdm


def ewc_loss(model, weight, estimated_fishers, estimated_means):
    losses = []
    for param_name, param in model.named_parameters():
        if "backbone" in param_name:
            estimated_mean = estimated_means[param_name]
            estimated_fisher = estimated_fishers[param_name]
            losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
    return (weight / 2) * sum(losses)


def estimate_ewc_params(model, triplet,loader):
    estimated_mean = {}
    estimated_fisher = {}

    for param_name, param in model.named_parameters():
        if "backbone" in param_name:
            estimated_mean[param_name] = param.data.clone()
            estimated_fisher[param_name] = torch.zeros_like(param)

    for it, (images, sketches,neg,label) in enumerate(tqdm(loader)):
        images = images.cuda()
        sketches = sketches.cuda()
        neg = neg.cuda()

        img_fea = model(images,'im')
        skt_fea = model(sketches,'sk')
        neg_fea = model(neg,'im')
        loss = triplet(skt_fea,img_fea,neg_fea)
        loss.backward()

        for param_name, param in model.named_parameters():
            if param.grad is not None:
                estimated_fisher[param_name] += param.grad.data ** 2 / len(loader)

    estimated_fisher = {n: p for n, p in estimated_fisher.items()}
    return estimated_mean, estimated_fisher

