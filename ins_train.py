import os
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from model.pointnet2.pointnet2_paconv_seg import PointNet2INSSeg as Model
import logging
from utils.indoor3d_util import args
from utils.tools import AverageMeter
from utils.s3dis import S3DIS
from utils.discriminative_loss import DiscriminativeLoss


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


def train(train_loader, model, criterion, optimizer, epoch):
    loss_meter = AverageMeter()
    model.train()
    pbar = tqdm(train_loader)
    for i,(points,colors, _,ins_label) in enumerate(pbar):
        points = points.float().cuda(non_blocking=True)
        colors = colors.float().cuda(non_blocking=True)
        ins_label = ins_label.long().cuda(non_blocking=True)
        ins_output = model(torch.cat((points,colors),dim=2))
        loss = criterion(ins_output,ins_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(),points.size(0))
        pbar.set_description('loss_meter:{}'.format(loss_meter.avg))

    logger.info('Loss result at epoch [{}/{}]: /ins_loss{:.4f}'
                .format(epoch+1,args['epochs'],loss_meter.avg))
    return loss_meter.avg


if __name__ == '__main__':
    if args['train_gpu'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args['train_gpu'])
    if args['train_gpu'] is not None and len(args['train_gpu']) == 1:
        args['sync_bn'] = False
    logger = get_logger()
    writer = SummaryWriter(args['save_path'])
    model = Model(c=args['fea_dim'],use_xyz=args['use_xyz'],args=args).cuda()
    if args['sync_bn']:
        from utils.tools import convert_to_syncbn
        convert_to_syncbn(model)
    criterion = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5,norm=1, alpha=1.0, beta=1.0, gamma=0.001,usegpu=True)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=args['base_lr'],momentum=args['momentum'],weight_decay=args['weight_decay'])
    if args.get('lr_multidecay', False):
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[int(args['epochs'] * 0.4), int(args['epochs'] * 0.6),
                                                         int(args['epochs'] * 0.8)], gamma=args['multiplier'])
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    logger.info('---creating Model---')
    logger.info('Classes:{}'.format(args['classes']))
    logger.info(model)
    model = torch.nn.DataParallel(model, device_ids=args['train_gpu'])  # multi-GPU
    model.to('cuda')
    #----------------------------load_weight or resume-----------------------
    if args['weight']:
        if os.path.isfile(args['weight']):
            loaded_state_dict = torch.load(args['weight'])
            model_state_dict = model.state_dict()
            pretrained_dict = {
                k: v
                for k, v in loaded_state_dict.items()
                if k in model_state_dict and model_state_dict[k].shape == v.shape
            }
            model_state_dict.update(pretrained_dict)
            model.load_state_dict(model_state_dict, strict=True)
        else:
            logger.info('no weight found at'.format(args['weight']))

    train_data = S3DIS(split='train', data_root=args['NEW_DATA_PATH'], test_area=args['test_area'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args['batch_size'], shuffle=True,
                                               num_workers=args['train_workers'])

    #----------------------------train-----------------------------------
    for epoch in range(args['epochs']):
        loss_train = train(train_loader, model, criterion, optimizer, epoch)
        epoch_log = epoch + 1
        writer.add_scalar('ins_loss_train', loss_train, epoch_log)

        filename = args['save_path'] + '/{}_train_{}.pth'.format(args.get('test_area',5),epoch_log)
        logger.info('Best Model Saving checkpoint to {}'.format(filename))
        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, filename)

        scheduler.step()
    logger.info('Train Finish!')



