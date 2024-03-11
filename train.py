r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch
import os
import math

from model.HiPA import HiPA
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
import torch.nn.functional as F

import wandb

#os.environ["CUDA_VISIBLE_DEVICES"]='1,2'

def train(epoch, model, dataloader, optimizer, training,lamda):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1),lamda)
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


#DDP train
if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()
    local_rank = args.local_rank

    # wandb
    if args.use_wandb:
        wandb.init(
            project="few-shot",
            config={
                "model": "DCAMA",
                "learning_rate": args.lr,
                "batch_size": args.bsz,
                "backbone": args.backbone,
                "dataset": args.benchmark,
            }
        )

    # ddp backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # Model initialization
    model = HiPA(args.backbone, args.feature_extractor_path, False)
    device = torch.device("cuda", local_rank)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                find_unused_parameters=True)

    # Helper classes (for training) initialization
    optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
                            "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
    Evaluator.initialize()
    if local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    if local_rank == 0:
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.nepoch):
        dataloader_trn.sampler.set_epoch(epoch)
        lamda=0.5*(1+math.cos((epoch+1)/args.nepoch*5*math.pi))
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True,lamda=lamda)

        # evaluation
        if epoch == range(args.nepoch)[-1]:
            break
        if local_rank == 0:
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False,lamda=lamda)

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(model, epoch, val_miou)

            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
            Logger.info('\nbest_val_miou=%f\n' % best_val_miou)

        if args.use_wandb:
            wandb.log({"train_avg_loss": trn_loss, "train_miou": trn_miou, "train_fb_iou": trn_fb_iou,
                       "val_avg_loss": val_loss, "val_miou": val_miou, "val_fb_iou": val_fb_iou,})

    if args.use_wandb:
        wandb.finish()

    if local_rank == 0:
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')

#DP
# if __name__ == '__main__':
#
#     # Arguments parsing
#     args = parse_opts()
#     #local_rank = args.local_rank
#
#     # wandb
#     if args.use_wandb:
#         wandb.init(
#             project="few-shot",
#             config={
#                 "model": "DCAMA",
#                 "learning_rate": args.lr,
#                 "batch_size": args.bsz,
#                 "backbone": args.backbone,
#                 "dataset": args.benchmark,
#                 "fold": args.fold
#             }
#         )
#
#     # ddp backend initialization
#     # torch.distributed.init_process_group(backend='nccl')
#     # torch.cuda.set_device(local_rank)
#
#     # Model initialization
#     model = DCAMA(args.backbone, args.feature_extractor_path, False)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model, device_ids=[0, 1, 2], output_device=0)
#     else:
#         model = nn.DataParallel(model)
#     model.to(device)
#     # device = torch.device("cuda", local_rank)
#     # model.to(device)
#     # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
#     #                                             find_unused_parameters=True)
#
#     # Helper classes (for training) initialization
#     optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
#                             "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
#     Evaluator.initialize()
#     # if local_rank == 0:
#     Logger.initialize(args, training=True)
#     Logger.info('# available GPUs: %d' % torch.cuda.device_count())
#
#     # Dataset initialization
#     FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=False)
#     dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
#     # if local_rank == 0:
#     dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')
#
#     # Train
#     best_val_miou = float('-inf')
#     best_val_loss = float('inf')
#     for epoch in range(args.nepoch):
#         # dataloader_trn.sampler.set_epoch(epoch)
#         trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
#
#         # evaluation
#         #if local_rank == 0:
#         with torch.no_grad():
#             val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)
#
#         # Save the best model
#         if val_miou > best_val_miou:
#             best_val_miou = val_miou
#             Logger.save_model_miou(model, epoch, val_miou)
#
#         Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
#         Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
#         Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
#         Logger.tbd_writer.flush()
#
#         if args.use_wandb:
#             wandb.log({"train_avg_loss": trn_loss, "train_miou": trn_miou, "train_fb_iou": trn_fb_iou,
#                        "val_avg_loss": val_loss, "val_miou": val_miou, "val_fb_iou": val_fb_iou,})
#
#     if args.use_wandb:
#         wandb.finish()
#
#     #if local_rank == 0:
#     Logger.tbd_writer.close()
#     Logger.info('==================== Finished Training ====================')


