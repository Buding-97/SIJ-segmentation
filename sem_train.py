import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from model.pointnet2.pointnet2_paconv_seg import PointNet2SSGSeg as Model
from model.pointnet2.paconv import PAConv
import logging
from utils.indoor3d_util import args
from utils.tools import AverageMeter,intersectionAndUnionGPU
from utils.s3dis import S3DIS


def get_logger():
    logger = logging.getLogger("main-train-logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s")
    file_handler = logging.FileHandler(os.path.join(args['save_path'],'train.log'))
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def train(train_loader,model, criterion, optimizer, epoch, correlation_loss):
    loss_meter = AverageMeter()
    sem_loss_meter = AverageMeter()
    ins_loss_meter = AverageMeter()
    corr_loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    for i,(points,colors, sem_lable,_) in enumerate(tqdm(train_loader)):
        points = points.float().cuda(non_blocking=True)
        colors = colors.float().cuda(non_blocking=True)
        sem_lable = sem_lable.long().cuda(non_blocking=True)
        sem_output = model(torch.concat((points,colors),dim=2))
        sem_loss = criterion(sem_output,sem_lable)
        corr_loss = 0.0
        corr_loss_scale = args.get('correlation_loss_scale',10.0)
        if correlation_loss:
            for m in model.module.SA_modules.named_modules():
                if isinstance(m[-1],PAConv):
                    kernel_matrice, output_dim, m_dim = m[-1].weightbank, m[-1].output_dim, m[-1].m
                    new_kernel_matrice = kernel_matrice.view(-1, m_dim, output_dim).permute(1, 0, 2).reshape(m_dim, -1)
                    cost_matrice = torch.matmul(new_kernel_matrice, new_kernel_matrice.T) / torch.matmul(
                        torch.sqrt(torch.sum(new_kernel_matrice ** 2, dim=-1, keepdim=True)),
                        torch.sqrt(torch.sum(new_kernel_matrice.T ** 2, dim=0, keepdim=True)))
                    corr_loss += torch.sum(torch.triu(cost_matrice, diagonal=1) ** 2)
        loss = sem_loss+corr_loss_scale*corr_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sem_output = sem_output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(sem_output , sem_lable, args['classes'], args['ignore_label'])
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        loss_meter.update(loss.item(),points.size(0))
        sem_loss_meter.update(sem_loss.item(), points.size(0))
        corr_loss_meter.update(corr_loss.item() * corr_loss_scale if correlation_loss else corr_loss, points.size(0))
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Loss result at epoch [{}/{}]: loss/sem_loss/ins_loss/corr_loss {:.4f}/{:.4f}/{:.4f}/{:.4f}'
                .format(epoch+1,args['epochs'],loss_meter.avg,sem_loss_meter.avg,ins_loss_meter.avg,corr_loss_meter.avg))
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}'.format(epoch+1,args['epochs'],mIoU,mAcc,allAcc))
    return loss_meter.avg,mIoU,mAcc,allAcc


def validate(val_loader, model, criterion):
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (points, colors, sem_lable, _) in enumerate(tqdm(val_loader)):
            points = points.float().cuda(non_blocking=True)
            colors = colors.float().cuda(non_blocking=True)
            sem_lable = sem_lable.long().cuda(non_blocking=True)
            sem_output = model(torch.concat((points, colors), dim=2))
            sem_loss = criterion(sem_output,sem_lable)
            sem_output = sem_output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(sem_output, sem_lable, args['classes'], args['ignore_label'])
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            loss_meter.update(sem_loss.item(),points.size(0))
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args['classes']):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    if args['train_gpu'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args['train_gpu'])
    if args['train_gpu'] is not None and len(args['train_gpu']) == 1:
        args['sync_bn'] = False
    logger = get_logger()
    writer = SummaryWriter(args['save_path'])
    model = Model(c=args['fea_dim'],k=args['classes'],use_xyz=args['use_xyz'],args=args).cuda()
    if args['sync_bn']:
        from utils.tools import convert_to_syncbn
        convert_to_syncbn(model)
    sem_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr=args['base_lr'],momentum=args['momentum'],weight_decay=args['weight_decay'])
    if args.get('lr_multidecay',False):
        scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[int(args['epochs'] * 0.6), int(args['epochs'] * 0.8)],
                                             gamma=args['multiplier'])
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_epoch'], gamma=args['multiplier'])
    logger.info('---creating Model---')
    logger.info('Classes:{}'.format(args['classes']))
    logger.info(model)
    model = torch.nn.DataParallel(model,device_ids=args['train_gpu']) # multi-GPU
    #----------------------------load_weight or resume-----------------------
    if args['weight']:
        if os.path.isfile(args['weight']):
            logger.info('loading weight {}'.format(args['weight']))
            checkpoint = torch.load(args['weight'])
            model.load_state_dict(checkpoint['state_dict'])
            logger.info('loaded weight {}'.format(args['weight']))
        else:
            logger.info('no weight found at'.format(args['weight']))
    train_data = S3DIS(split='train', data_root=args['NEW_DATA_PATH'], test_area=args['test_area'])
    val_data = S3DIS(split='test', data_root=args['NEW_DATA_PATH'], test_area=args['test_area'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=args['train_workers'])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args['val_batch_size'], shuffle=False, num_workers=args['train_workers'])
    #----------------------------train-----------------------------------
    for epoch in range(args['epochs']):
        loss_train,mIou_train,mAcc_train,allAcc_train = train(train_loader, model, sem_criterion, optimizer, epoch, args.get('correlation_loss', False))
        epoch_log = epoch + 1
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIou_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train',allAcc_train, epoch_log)
        filename = args['save_path'] + '/{}_train_{}.pth'.format(args['test_area'],epoch_log)
        logger.info('Best Model Saving checkpoint to {}'.format(filename))
        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, filename)
        ######validate#########
        loss_val,mIou_val,mAcc_val,allAcc_val = validate(val_loader,model,sem_criterion)
        #######################
        scheduler.step()
    logger.info('Train Finish!')
