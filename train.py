import os
from model import mymodel
from tookits import utils
import argparse
import datetime
import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from datasets import Datasets, TestDatasets
from tqdm import tqdm
from tookits.calculate_prototype import Prototype
from tookits.ewc import estimate_ewc_params, ewc_loss
from valid import valid
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

loss_cn = torch.nn.CrossEntropyLoss().cuda()


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--batch_size', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=5, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")

    # Misc
    parser.add_argument('--output_dir', default="outputs", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--num_workers', default=6, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_experts', default=6, type=int)

    # datasets
    parser.add_argument("--root", default="../datasets/", type=str)
    # parser.add_argument("--root", default="../../hy-tmp/", type=str)
    # parser.add_argument("--datasets", default="QuickDraw", type=str, choices=["Sketchy", "QuickDraw", "TUBerlin"])
    parser.add_argument("--datasets", default="Sketchy", type=str, choices=["Sketchy", "QuickDraw", "TUBerlin"])
    # parser.add_argument("--datasets",default="TUBerlin",type=str,choices=["Sketchy","QuickDraw","TUBerlin"])
    return parser


def train_session(args,stage):
    cudnn.benchmark = True
    utils.fix_random_seeds(1234)
    # ============ preparing data ... ============
    img_txt = f"datasets_list/{args.datasets}/{stage}_img.txt"
    skt_txt = f"datasets_list/{args.datasets}/{stage}_skt_train.txt"

    length = 10000
    if args.datasets == "QuickDraw":
        length = 30000
    dataset = Datasets(root = args.root,datasets = args.datasets, img_txt = img_txt, skt_txt = skt_txt,length = length)
    data_loader = DataLoader(
        dataset,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True,drop_last=True,
    )
    # ============ building student and teacher networks ... ============
    model = mymodel(dataset.num_class,num_experts=args.num_experts)
    model_old = mymodel(dataset.num_class,num_experts=args.num_experts)

    if stage == 1:
        #load vision transformer pre_train_weights on session 1
        model.load_state_dict(torch.load(f'{args.output_dir}/baseline/{args.datasets}_stage{stage}.pth')['model'],strict=False)
        prototype_skt = torch.zeros(dataset.num_class, 384)
        prototype_img = torch.zeros(dataset.num_class, 384)
        known_classes = 0
    else:
        parameters = torch.load(f'{args.output_dir}/{args.datasets}_stage{stage-1}_moe.pth')
        model.load_state_dict(parameters['model'],strict=False)
        model_old.load_state_dict(parameters['model'],strict=False)
        prototype_skt = parameters["skt"].cpu()
        prototype_img = parameters["img"].cpu()
        known_classes = prototype_skt.shape[0]
        prototype_skt_new = torch.zeros(dataset.num_class, 384)
        prototype_img_new = torch.zeros(dataset.num_class, 384)
        prototype_skt = torch.cat([prototype_skt_new,prototype_skt],dim=0).cuda()
        prototype_img = torch.cat([prototype_img_new,prototype_img],dim=0).cuda()

    prototype = Prototype(prototype_img,prototype_skt).cuda()

    for param in model_old.parameters():
        param.requires_grad = False

    print(f"Stage:{stage}, Known classes:{known_classes}")

    model.reset()
    model = model.cuda()
    model_old = model_old.cuda()
    triplet = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(model.parameters())  # to use with ViTs
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size/ 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    start_time = time.time()
    estimated_mean, estimated_fisher = None, None
    if stage>1:
        img_txt_old = f"datasets_list/{args.datasets}/{stage-1}_img.txt"
        skt_txt_old  = f"datasets_list/{args.datasets}/{stage-1}_skt_train.txt"

        dataset_old = Datasets(root=args.root, datasets=args.datasets, img_txt=img_txt_old, skt_txt=skt_txt_old)
        data_loader_old = DataLoader(
            dataset_old, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        estimated_mean, estimated_fisher = estimate_ewc_params(model,triplet,data_loader_old)

    for epoch in range(args.epochs):

        train_one_epoch(model, data_loader, optimizer, lr_schedule, wd_schedule,
            epoch, fp16_scaler,triplet,model_old,prototype,dataset.num_class,stage,estimated_mean, estimated_fisher)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    save_dict = {
        'model': model.state_dict(),
        "skt":prototype.center_skt.cpu(),
        "img":prototype.center_img.cpu()
    }
    torch.save(save_dict, f'{args.output_dir}/{args.datasets}_stage{stage}_moe.pth')

    img_loader = TestDatasets(args.root, args.datasets, m='img', stage=stage)
    skt_loader = TestDatasets(args.root, args.datasets, m='skt', stage=stage)

    img_loader = DataLoader(img_loader, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    skt_loader = DataLoader(skt_loader, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    mAP, Prec = valid(img_loader, skt_loader, model, 100, True)
    print(f"STAGE {stage}", mAP, Prec)
    return mAP, Prec

def train_one_epoch(model, data_loader, optimizer, lr_schedule, wd_schedule,
            epoch, fp16_scaler,triplet,model_old,prototype:Prototype,n,stage,estimated_mean, estimated_fisher):

    l1 = utils.AverageMeter()
    l2 = utils.AverageMeter()
    l3 = utils.AverageMeter()
    l4 = utils.AverageMeter()
    l5 = utils.AverageMeter()

    for it, (images, sketches,neg,label) in enumerate(tqdm(data_loader)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        images = images.cuda()
        sketches = sketches.cuda()
        neg = neg.cuda()
        label = label.cuda()
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            img_fea = model(images,"im")
            skt_fea = model(sketches,"sk")
            neg_fea = model(neg,"im")
            img_p = model.p_head(img_fea)
            skt_p = model.p_head(skt_fea)

            tri_loss = triplet(skt_fea,img_fea,neg_fea)

            cls_loss = loss_cn(model.fc(img_fea),label) + loss_cn(model.fc(skt_fea),label)

            ca_loss = prototype(img_p,label,'img') + prototype(skt_p,label,'skt')

            loss = tri_loss + cls_loss + ca_loss

            if stage>1:
                skt_old = model_old(sketches,'sk').detach()
                img_old = model_old(images,'im').detach()

                # matrix_new = skt_fea @ img_fea.T
                # matrix_old = skt_old @ img_old.T

                l2_loss = ewc_loss(model, 1000, estimated_fisher, estimated_mean)
                # l2_loss = F.kl_div(skt_fea,skt_old)+F.kl_div(img_fea,img_old)
                l3.update(l2_loss.item())
                loss = loss + l2_loss

                img_old_p = model_old.p_head(img_old)
                skt_old_p = model_old.p_head(skt_old)

                shift_loss = prototype.update_shift(img_old_p,img_p,skt_old_p,skt_p,n,len(data_loader))
                l5.update(shift_loss.item())

                loss = loss + shift_loss

            l1.update(tri_loss.item())
            l2.update(cls_loss.item())
            l4.update(ca_loss.item())

        # student update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        torch.cuda.synchronize()
    if stage ==1:
        print(f"stage{stage},epoch:{epoch},TRI LOSS:{l1.avg:.6f},CLS LOSS:{l2.avg:.6f},ca_loss{l4.avg:.6f}")
    else:
        # print(f"stage{stage},epoch:{epoch},TRI LOSS:{l1.avg},CLS LOSS:{l2.avg},ca_loss{l4.avg},l2_loss{l3.avg},s_loss:{l5.avg}")
        print(f"stage{stage},epoch:{epoch},TRI LOSS:{l1.avg:.6f},CLS LOSS:{l2.avg:.6f},ca_loss{l4.avg:.6f},s_loss:{l5.avg:.6f},dis:{l3.avg:.6f}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    m = []
    p = []

    stage = 5
    if args.datasets == "TUBerlin":
        stage = 10

    print(args.datasets)
    for i in range(stage):
        mAP, Prec= train_session(args,i+1)
        m.append(mAP)
        p.append(Prec)

    for i in range(len(m)):
        print("stage",i+1,m[i],p[i])

    import numpy as np
    print(np.mean(np.array(m)), np.mean(np.array(p)))
