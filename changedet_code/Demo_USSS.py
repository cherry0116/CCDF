import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import os
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np
import cv2
from tqdm import tqdm
import gc
from PIL import Image
import time
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable

from Module import *
from data_utils import *
from metrics import Evaluator
from CommonFunc import *
from Loss import *
import argparse
import ast
import random
import logging

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# code for unsupervised change detection
def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_num_epochs_G', type=int, default=30)
    parser.add_argument('--init_num_epochs_S', type=int, default=30)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--perception_weight', type=float, default=0.4)
    parser.add_argument('--l1_weight', type=float, default=0.65)
    parser.add_argument('--ssim_weight', type=float, default=0)
    parser.add_argument('--consist_weight', type=float, default=0.7)
    parser.add_argument('--perception_perBand', type=bool, default=True)
    parser.add_argument('--write_color', type=bool, default=True)
    parser.add_argument('--discriminator_continuous', type=bool, default=True)
    parser.add_argument('--perception_layer', type=int, default=1)

    parser.add_argument('--data_name', type=str, default='HY')
    parser.add_argument('--dir', type=str, default='../dataset/')    
    parser.add_argument('--ImageXName', type=str, default='T1.tif')
    parser.add_argument('--ImageYName', type=str, default='T2.tif')
    parser.add_argument('--RefName', type=str, default='ref.tif')
    parser.add_argument('--outFileName', type=str, default='ChangeDensity')

    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--prob_thresh', type=float, default=0.5)
    parser.add_argument('--patch_size', type=str, default='(224, 224)')
    parser.add_argument('--overlap_padding', type=str, default='(12, 12)')
    parser.add_argument('--aug_flag', type=str, default='Y')
    parser.add_argument('--cycle_flag', type=str, default='N')
    parser.add_argument('--consist_flag', type=str, default='Y')

    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    return args

def main():
    train_opt = parse_opt()
    time_now = time.strftime("%Y.%m.%d_%H%M%S", time.localtime())
    log_dir = '%s/%s'%(train_opt.log_dir, time_now + '_' + train_opt.data_name)
    os.makedirs(log_dir, exist_ok = True)

    # read the image to calculate the mean/std
    ImgXPath = os.path.join(train_opt.dir, train_opt.data_name, train_opt.ImageXName)
    ImgYPath = os.path.join(train_opt.dir, train_opt.data_name, train_opt.ImageYName)
    RefPath = os.path.join(train_opt.dir, train_opt.data_name, train_opt.RefName)

    # back up file
    backup_lists = []
    for _ in  os.listdir(os.getcwd()):
        if _.endswith('.py'):
            backup_lists.append(_)
    backup_code(backup_lists, log_dir)

    # the label to indicate change/non-change in reference (ground truth) and prediction change map for convenience
    gt_map = [1, 2]
    pre_map = [0, 1]
    # accuracy evaluation
    acc = Evaluator(num_class=len(gt_map))

    if train_opt.evaluate:
        pattern_item = 'complete'
        log_pattern_dir = os.path.join(log_dir, pattern_item)
        OutPath = os.path.join(log_dir, pattern_item, train_opt.outFileName)
        OutColorPath = os.path.join(log_dir, pattern_item, train_opt.outFileName)
        
        os.makedirs(log_pattern_dir, exist_ok = True)
        resume_statsXPath, resume_statsYPath = os.path.join(train_opt.resume, 'T1_stats.txt'), os.path.join(train_opt.resume, 'T2_stats.txt')
        meanX, stdX, meanY, stdY = Dataset_meanstd(resume_statsXPath, resume_statsYPath, '')
        scaler = NORMALIZE(meanX, stdX, meanY, stdY)

        G_modelPath = os.path.join(train_opt.resume, pattern_item, 'GModel_{}.pkl'.format(pattern_item))
        S_modelPath = os.path.join(train_opt.resume, pattern_item, 'SModel_{}.pkl'.format(pattern_item))      
        dataset = GDALDataset(ImgXPath, ImgYPath, refPath=RefPath, outPath=OutPath+'_%s.tif'%(pattern_item), enhance=scaler, \
                patch_size=ast.literal_eval(train_opt.patch_size), overlap_padding=ast.literal_eval(train_opt.overlap_padding),\
                aug_flag='Y', stage='test')

        # model
        netS = Segmentor(n_channels=dataset.size()[2], bilinear=True)
        netS.to(device)

        netG = Generator(n_channels=dataset.size()[2])
        netG.to(device)
        netG.load_state_dict(torch.load(G_modelPath))
        netS.load_state_dict(torch.load(S_modelPath))       
        eval_result(train_opt, netS, netG, dataset, acc, log_pattern_dir, gt_map, pre_map,  OutColorPath + '_acc_color_%s.tif'%(pattern_item))
        exit()

    # cal mean and std
    dataset = GDALDataset(ImgXPath, ImgYPath, patch_size = (220, 220), overlap_padding = (0, 0))
    statsPath1 = os.path.join(log_dir, '{}_stats.txt'.format(train_opt.ImageXName.split('.')[0]))
    statsPath2 = os.path.join(log_dir, '{}_stats.txt'.format(train_opt.ImageYName.split('.')[0]))
    meanX, stdX, meanY, stdY = Dataset_meanstd(statsPath1, statsPath2, dataset)
    del dataset

    # normalize the input image
    scaler = NORMALIZE(meanX, stdX, meanY, stdY)
    pattern_item = 'complete'
    log_pattern_dir = os.path.join(log_dir, pattern_item)
    OutPath = os.path.join(log_dir, pattern_item, train_opt.outFileName)
    OutColorPath = os.path.join(log_dir, pattern_item, train_opt.outFileName)
        
    os.makedirs(log_pattern_dir, exist_ok = True)
    # train
    dataset_train = GDALDataset(ImgXPath, ImgYPath, refPath=RefPath, outPath=OutPath+'_%s.tif'%(pattern_item), enhance=scaler, \
            patch_size=ast.literal_eval(train_opt.patch_size), overlap_padding=ast.literal_eval(train_opt.overlap_padding),\
            aug_flag='Y', stage='train', consist_flag=train_opt.consist_flag)
    netS, netG = transform_onepattern(train_opt, dataset_train, acc, log_pattern_dir, gt_map, pre_map, pattern_item, meanY, stdY)

    # test
    dataset_test = GDALDataset(ImgXPath, ImgYPath, refPath=RefPath, outPath=OutPath+'_%s.tif'%(pattern_item), enhance=scaler, \
            patch_size=ast.literal_eval(train_opt.patch_size), overlap_padding=ast.literal_eval(train_opt.overlap_padding),\
            aug_flag='N', stage='test', consist_flag='N')
    eval_result(train_opt, netS, netG, dataset_test, acc, log_dir, gt_map, pre_map, OutColorPath + '_acc_color_%s.tif'%(pattern_item))

def save_gen_image(train_opt, dataset, netG, meanY, stdY,  log_dir, pattern_item, stage):
    pad = dataset.overlap_padding
    dataloader_item = DataLoader(dataset, batch_size=train_opt.batch_size, shuffle=False)

    ReconstructPath = log_dir + '/reconstruct_%s_%s.tif'%(stage, pattern_item)
    xsize, ysize, nband = dataset.size()

    driver = dataset.imgDS_y.GetDriver()
    outDS = driver.Create(ReconstructPath, xsize, ysize, nband, dataset.imgDS_y.GetRasterBand(1).DataType)
    outDS.SetGeoTransform(dataset.imgDS_y.GetGeoTransform())
    outDS.SetProjection(dataset.imgDS_y.GetProjection())

    with torch.no_grad():
        for data_array in dataloader_item:
            x = data_array[0]
            item = data_array[2]

            x = x.to(device)
            y_fake = netG(x)

            for ns in range(x.size(0)):
                write_reconstruct = y_fake[ns].cpu().numpy()
                # UNNORMALIZE
                xitem_count, yitem_count = dataset.patch_count()
                item_x = math.floor(item[ns].numpy() / yitem_count)
                item_y = item[ns].numpy() % yitem_count

                slice, slice_read, slice_write = dataset.slice_assign(item_x, item_y)
                for b in range(outDS.RasterCount):
                    outBand = outDS.GetRasterBand(b + 1)
                    outBand.WriteArray(write_reconstruct[b, pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]] * stdY[b] + meanY[b], slice[0], slice[1])

def save_seg_image(train_opt, dataset, netG, netS, log_dir, pattern_item, stage):
    pad = dataset.overlap_padding
    dataloader_item = DataLoader(dataset, batch_size=train_opt.batch_size, shuffle=False)

    SegvisPath = log_dir + '/cls_%s_%s.png'%(stage, pattern_item)
    xsize, ysize, nband = dataset.size()

    cls_result = np.zeros((xsize, ysize), dtype = np.uint8)

    with torch.no_grad():
        for data_array in dataloader_item:
            x = data_array[0]
            y = data_array[1]
            item = data_array[2]

            x = x.to(device)
            y = y.to(device)

            cmap = netS(x, y)
            cmask = torch.zeros_like(cmap)
            cmask[cmap > train_opt.prob_thresh] = 255

            for ns in range(x.size(0)):
                change_mask = cmask[ns].cpu().numpy()
                xitem_count, yitem_count = dataset.patch_count()
                item_x = math.floor(item[ns].numpy() / yitem_count)
                item_y = item[ns].numpy() % yitem_count

                slice, slice_read, slice_write = dataset.slice_assign(item_x, item_y)
                cls_result[slice[1]:slice[1]+slice[2], slice[0]:slice[0]+slice[3]] = change_mask[0, pad[0]:pad[0] + slice[2], pad[1]:pad[1] + slice[3]]
        img = Image.fromarray(cls_result)
        img.save(SegvisPath)

def transform_onepattern(train_opt, dataset, acc, log_dir, gt_map, pre_map, pattern_item, meanY, stdY):
    # tensorboard to record experiment
    f_log = open(log_dir + '/%s.log'%(pattern_item), 'a')
    print('Arguments:\n')
    for k in train_opt.__dict__.keys():
        print('%s : %s '%(k, str(train_opt.__dict__[k])))
        f_log.write('%s : %s \n'%(k, str(train_opt.__dict__[k])))
    writer = SummaryWriter(log_dir=log_dir, comment='USSS_{}'.format(pattern_item))      

    # data loader
    train_dataloader = DataLoader(dataset, batch_size=train_opt.batch_size, shuffle=True)
    pad = dataset.overlap_padding

    # model
    netS = Segmentor(n_channels=dataset.size()[2], bilinear=True)
    netS.to(device)

    netG_x2y = Generator(n_channels=dataset.size()[2])
    netG_x2y.to(device)

    netS.train()
    netG_x2y.train()

    criterion = CNetLoss(channel=dataset.size()[2], perception_layer=train_opt.perception_layer, perception_perBand=train_opt.perception_perBand)
    criterion.to(device)
    optimizerS = torch.optim.Adam(netS.parameters(), lr=0.0002, betas=(0.9, 0.99))
    optimizerG_x2y = torch.optim.Adam(netG_x2y.parameters(), lr=0.0002, betas=(0.9, 0.99))

    if train_opt.cycle_flag == 'Y':
        netG_y2x = Generator(n_channels=dataset.size()[2])
        netG_y2x.to(device)
        netG_y2x.train()
        optimizerG_y2x = torch.optim.Adam(netG_y2x.parameters(), lr=0.0002, betas=(0.9, 0.99))
    
    print('Start Initial Generator Training')
    f_log.write('\nStart Initial Generator Training\n')
    with torch.enable_grad():
        for i in range(train_opt.init_num_epochs_G):
            NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver, ssim_loss_aver = 0, 0, 0, 0, 0
            # warm-up strategy
            adjust_learning_rate(optimizerG_x2y, i, lr_start=1e-5, lr_max=3e-4, lr_warm_up_epoch=10, lr_sustain_epochs=10)

            if train_opt.cycle_flag == 'Y':
                NetLoss_aver_y2x, generator_loss_aver_y2x, l1_loss_aver_y2x, perception_loss_aver_y2x, ssim_loss_aver_y2x = 0, 0, 0, 0, 0
                NetLoss_aver_y2x2y, generator_loss_aver_y2x2y, l1_loss_aver_y2x2y, perception_loss_aver_y2x2y, ssim_loss_aver_y2x2y = 0, 0, 0, 0, 0
                NetLoss_aver_x2y2x, generator_loss_aver_x2y2x, l1_loss_aver_x2y2x, perception_loss_aver_x2y2x, ssim_loss_aver_x2y2x = 0, 0, 0, 0, 0
                adjust_learning_rate(optimizerG_y2x, i, lr_start=1e-5, lr_max=3e-4, lr_warm_up_epoch=10, lr_sustain_epochs=10)

            process_num = 0
            for data_array in train_dataloader:
                time_start = time.time()
                # Update G network:

                x = data_array[0]
                y = data_array[1]

                x = x.to(device)
                y = y.to(device)  

                # x to y
                optimizerG_x2y.zero_grad()
                y_fake = netG_x2y(x)
                cmap = torch.zeros((x.size()[0], 1, x.size()[2], x.size()[3]))
                cmap = cmap.to(device)

                generator_loss, l1_loss, perception_loss, ssim_loss = criterion(y, y_fake, cmap)
                Loss = generator_loss + train_opt.perception_weight * perception_loss + train_opt.ssim_weight * ssim_loss

                Loss.backward()
                optimizerG_x2y.step()

                NetLoss_aver += Loss * x.size(0) / len(dataset)
                generator_loss_aver += generator_loss * x.size(0) / len(dataset)
                l1_loss_aver += l1_loss * x.size(0) / len(dataset)
                perception_loss_aver += perception_loss * x.size(0) / len(dataset)
                ssim_loss_aver += ssim_loss * x.size(0) / len(dataset)

                if train_opt.cycle_flag == 'Y':
                    # y to x
                    optimizerG_y2x.zero_grad()
                    x_fake = netG_y2x(y)
                    cmap = torch.zeros((y.size()[0], 1, y.size()[2], y.size()[3]))
                    cmap = cmap.to(device)

                    generator_loss_y2x, l1_loss_y2x, perception_loss_y2x, ssim_loss_y2x = criterion(x, x_fake, cmap)
                    Loss_y2x = generator_loss_y2x + train_opt.perception_weight * perception_loss_y2x + train_opt.ssim_weight * ssim_loss_y2x

                    Loss_y2x.backward()
                    optimizerG_y2x.step()

                    NetLoss_aver_y2x += Loss_y2x * x.size(0) / len(dataset)
                    generator_loss_aver_y2x += generator_loss_y2x * x.size(0) / len(dataset)
                    l1_loss_aver_y2x += l1_loss_y2x * x.size(0) / len(dataset)
                    perception_loss_aver_y2x += perception_loss_y2x * x.size(0) / len(dataset)
                    ssim_loss_aver_y2x += ssim_loss_y2x * x.size(0) / len(dataset)

                    # y to x to y
                    optimizerG_x2y.zero_grad()
                    y_fake_fake = netG_x2y(netG_y2x(y))
                    cmap = torch.zeros((y.size()[0], 1, y.size()[2], y.size()[3]))
                    cmap = cmap.to(device)

                    generator_loss_y2x2y, l1_loss_y2x2y, perception_loss_y2x2y, ssim_loss_y2x2y = criterion(y, y_fake_fake, cmap)
                    Loss_y2x2y = generator_loss_y2x2y + train_opt.perception_weight * perception_loss_y2x2y + train_opt.ssim_weight * ssim_loss_y2x2y

                    Loss_y2x2y.backward()
                    optimizerG_x2y.step()

                    NetLoss_aver_y2x2y += Loss_y2x2y * x.size(0) / len(dataset)
                    generator_loss_aver_y2x2y += generator_loss_y2x2y * x.size(0) / len(dataset)
                    l1_loss_aver_y2x2y += l1_loss_y2x2y * x.size(0) / len(dataset)
                    perception_loss_aver_y2x2y += perception_loss_y2x2y * x.size(0) / len(dataset)
                    ssim_loss_aver_y2x2y += ssim_loss_y2x2y * x.size(0) / len(dataset)

                    # x to y to x
                    optimizerG_y2x.zero_grad()
                    x_fake_fake = netG_y2x(netG_x2y(x))
                    cmap = torch.zeros((x.size()[0], 1, x.size()[2], x.size()[3]))
                    cmap = cmap.to(device)

                    generator_loss_x2y2x, l1_loss_x2y2x, perception_loss_x2y2x, ssim_loss_x2y2x = criterion(x, x_fake_fake, cmap)
                    Loss_x2y2x = generator_loss_x2y2x + train_opt.perception_weight * perception_loss_x2y2x + train_opt.ssim_weight * ssim_loss_x2y2x

                    Loss_x2y2x.backward()
                    optimizerG_y2x.step()

                    NetLoss_aver_x2y2x += Loss_x2y2x * x.size(0) / len(dataset)
                    generator_loss_aver_x2y2x += generator_loss_x2y2x * x.size(0) / len(dataset)
                    l1_loss_aver_x2y2x += l1_loss_x2y2x * x.size(0) / len(dataset)
                    perception_loss_aver_x2y2x += perception_loss_x2y2x * x.size(0) / len(dataset)
                    ssim_loss_aver_x2y2x += ssim_loss_x2y2x * x.size(0) / len(dataset)
                   
                process_num += x.size()[0]
                time_end = time.time()
                time_per_iter = (time_end - time_start) / x.size()[0] * len(dataset)
                time_remaining = time_per_iter * (
                        (train_opt.init_num_epochs_G - 1 - i) + (1 - process_num / len(dataset)))
                time_desc_per = time_show(time_per_iter)
                time_desc = time_show(time_remaining)

                print('\rProcessing batch: {}/{}; Processing speed per iter: {}; Processing time remaining: {}'.format(
                        process_num, len(dataset), time_desc_per, time_desc), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Epochs: {}/{}, NetLoss: {:.4f}, generator_loss: {:.4f}, l1_loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}'.format(
                    i + 1, train_opt.init_num_epochs_G, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))
            f_log.write(
                'Epochs: {}/{}, NetLoss: {:.4f}, generator_loss: {:.4f}, l1_loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}\n'.format(
                    i + 1, train_opt.init_num_epochs_G, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))
            
            writer.add_scalar('NetLoss', NetLoss_aver, i)
            writer.add_scalar('generator_loss', generator_loss_aver, i)
            writer.add_scalar('l1_loss', l1_loss_aver, i)
            writer.add_scalar('perception_loss', perception_loss_aver, i)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i)

            if train_opt.cycle_flag == 'Y':
                # y to x
                print(
                    'Epochs: {}/{}, NetLoss_y2x: {:.4f}, generator_loss_y2x: {:.4f}, l1_loss_y2x: {:.4f}, perception_loss_y2x:{:.4f}, ssim_loss_y2x:{:.4f}'.format(
                    i + 1, train_opt.init_num_epochs_G, NetLoss_aver_y2x, generator_loss_aver_y2x, l1_loss_aver_y2x, perception_loss_aver_y2x,
                    ssim_loss_aver_y2x))
                f_log.write(
                    'Epochs: {}/{}, NetLoss_y2x: {:.4f}, generator_loss_y2x: {:.4f}, l1_loss_y2x: {:.4f}, perception_loss_y2x:{:.4f}, ssim_loss_y2x:{:.4f}\n'.format(
                    i + 1, train_opt.init_num_epochs_G, NetLoss_aver_y2x, generator_loss_aver_y2x, l1_loss_aver_y2x, perception_loss_aver_y2x,
                    ssim_loss_aver_y2x))
            
                writer.add_scalar('NetLoss_y2x', NetLoss_aver_y2x, i)
                writer.add_scalar('generator_loss_y2x', generator_loss_aver_y2x, i)
                writer.add_scalar('l1_loss_y2x', l1_loss_aver_y2x, i)
                writer.add_scalar('perception_loss_y2x', perception_loss_aver_y2x, i)
                writer.add_scalar('ssim_loss_y2x', ssim_loss_aver_y2x, i)

                # y to x to y
                print(
                    'Epochs: {}/{}, NetLoss_y2x2y: {:.4f}, generator_loss_y2x2y: {:.4f}, l1_loss_y2x2y: {:.4f}, perception_loss_y2x2y:{:.4f}, ssim_loss_y2x2y:{:.4f}'.format(
                    i + 1, train_opt.init_num_epochs_G, NetLoss_aver_y2x2y, generator_loss_aver_y2x2y, l1_loss_aver_y2x2y, perception_loss_aver_y2x2y,
                    ssim_loss_aver_y2x2y))
                f_log.write(
                    'Epochs: {}/{}, NetLoss_y2x2y: {:.4f}, generator_loss_y2x2y: {:.4f}, l1_loss_y2x2y: {:.4f}, perception_loss_y2x2y:{:.4f}, ssim_loss_y2x2y:{:.4f}\n'.format(
                    i + 1, train_opt.init_num_epochs_G, NetLoss_aver_y2x2y, generator_loss_aver_y2x2y, l1_loss_aver_y2x2y, perception_loss_aver_y2x2y,
                    ssim_loss_aver_y2x2y))
            
                writer.add_scalar('NetLoss_y2x2y', NetLoss_aver_y2x2y, i)
                writer.add_scalar('generator_loss_y2x2y', generator_loss_aver_y2x2y, i)
                writer.add_scalar('l1_loss_y2x2y', l1_loss_aver_y2x2y, i)
                writer.add_scalar('perception_loss_y2x2y', perception_loss_aver_y2x2y, i)
                writer.add_scalar('ssim_loss_y2x2y', ssim_loss_aver_y2x2y, i)

                # x to y to x
                print(
                    'Epochs: {}/{}, NetLoss_x2y2x: {:.4f}, generator_loss_x2y2x: {:.4f}, l1_loss_x2y2x: {:.4f}, perception_loss_x2y2x:{:.4f}, ssim_loss_x2y2x:{:.4f}\n'.format(
                    i + 1, train_opt.init_num_epochs_G, NetLoss_aver_x2y2x, generator_loss_aver_x2y2x, l1_loss_aver_x2y2x, perception_loss_aver_x2y2x,
                    ssim_loss_aver_x2y2x))
                f_log.write(
                    'Epochs: {}/{}, NetLoss_x2y2x: {:.4f}, generator_loss_x2y2x: {:.4f}, l1_loss_x2y2x: {:.4f}, perception_loss_x2y2x:{:.4f}, ssim_loss_x2y2x:{:.4f}\n\n'.format(
                    i + 1, train_opt.init_num_epochs_G, NetLoss_aver_x2y2x, generator_loss_aver_x2y2x, l1_loss_aver_x2y2x, perception_loss_aver_x2y2x,
                    ssim_loss_aver_x2y2x))
            
                writer.add_scalar('NetLoss_x2y2x', NetLoss_aver_x2y2x, i)
                writer.add_scalar('generator_loss_x2y2x', generator_loss_aver_x2y2x, i)
                writer.add_scalar('l1_loss_x2y2x', l1_loss_aver_x2y2x, i)
                writer.add_scalar('perception_loss_x2y2x', perception_loss_aver_x2y2x, i)
                writer.add_scalar('ssim_loss_x2y2x', ssim_loss_aver_x2y2x, i)
          
    # save reconstruct image
    save_gen_image(train_opt, dataset, netG_x2y, meanY, stdY, log_dir, 'complete', 'Gen')

    print('Start Initial Segmentor Training')
    f_log.write('Start Initial Segmentor Training\n')
    with torch.enable_grad():
        for i in range(train_opt.init_num_epochs_S):
            NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver, ssim_loss_aver = 0, 0, 0, 0, 0
            adjust_learning_rate(optimizerS, i, lr_start=1e-5, lr_max=3e-4, lr_warm_up_epoch=10, lr_sustain_epochs=10)
            if train_opt.consist_flag == 'Y':
                consist_loss_aver = 0

            acc.reset()

            process_num = 0
            for data_array in train_dataloader:
                time_start = time.time()
    
                x = data_array[0]
                y = data_array[1]
                item = data_array[2]
                ref = data_array[3]

                x = x.to(device)
                y = y.to(device)
                y_fake = netG_x2y(x)
                cmap = netS(x, y)
                         
                generator_loss, l1_loss, perception_loss, ssim_loss = criterion(y, y_fake, cmap)
    
                NetLoss = generator_loss + train_opt.l1_weight * l1_loss + train_opt.perception_weight * perception_loss + train_opt.ssim_weight * ssim_loss

                if train_opt.aug_flag == 'Y' and train_opt.consist_flag == 'Y':
                    x_ori = data_array[4]
                    y_ori = data_array[5]
                    ori_flag = data_array[7]
                    after_flag = data_array[8]

                    recover_flag = ori_flag + after_flag
                    x_ori = x_ori.to(device)
                    y_ori = y_ori.to(device)
                    y_ori_fake = netG_x2y(x_ori).detach()
                    cmap_ori = netS(x_ori, y_ori).detach()
                    cmap_consist = torch.zeros_like(cmap_ori)

                    for index, cmap_ori_item in enumerate(cmap_ori):
                        cmap_consist[index] = dataset.augmentation_image(cmap_ori_item, recover_flag[index])
                    consist_loss = nn.L1Loss()(cmap, cmap_consist)
                    NetLoss = NetLoss + train_opt.consist_weight * consist_loss

                optimizerS.zero_grad()
                NetLoss.backward()
    
                optimizerS.step()
    
                NetLoss_aver += NetLoss * x.size(0) / len(dataset)
                generator_loss_aver += generator_loss * x.size(0) / len(dataset)
                l1_loss_aver += l1_loss * x.size(0) / len(dataset)
                perception_loss_aver += perception_loss * x.size(0) / len(dataset)
                ssim_loss_aver += ssim_loss * x.size(0) / len(dataset)
                if train_opt.consist_flag == 'Y':
                    consist_loss_aver += consist_loss * x.size(0) / len(dataset)
    
                cmask = torch.zeros_like(cmap)
                cmask[cmap > train_opt.prob_thresh] = 1
                for ns in range(x.size(0)):
                    change_mask = cmask[ns][0]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns][0].numpy()
    
                    item_x = math.floor(item[ns].numpy() / dataset.patch_count()[1])
                    item_y = item[ns].numpy() % dataset.patch_count()[1]
                    slice, _, _ = dataset.slice_assign(item_x, item_y)
    
                    # accuracy evaluation only with the centering region of the patch
                    acc.add_batch_map(ref_mask[pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16), change_mask[pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16), gt_map, pre_map)
    
                process_num += x.size()[0]
                time_end = time.time()
                time_per_iter = (time_end - time_start) / x.size()[0] * len(dataset)
                time_remaining = time_per_iter * (
                            (train_opt.num_epochs - 1 - i) + (1 - process_num / len(dataset)))
                time_desc_per = time_show(time_per_iter)
                time_desc = time_show(time_remaining)
    
                print('\rProcessing batch: {}/{}; Processing speed per iter: {}; Processing time remaining: {}'.format(
                        process_num, len(dataset), time_desc_per, time_desc), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Epochs: {}/{}, NetLoss: {:.4f}, generator_loss: {:.4f}, l1_loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}'.format(
                    i + 1, train_opt.init_num_epochs_S, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))

            print(
                'Epochs: {}/{}, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                    i + 1, train_opt.init_num_epochs_S, acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                    acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                    acc.Mean_Intersection_over_Union()[1]))

            f_log.write('Epochs: {}/{}, NetLoss: {:.4f}, generator_loss: {:.4f}, l1_loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}\n'.format(
                    i + 1, train_opt.init_num_epochs_S, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))

            f_log.write('Epochs: {}/{}, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}\n'.format(
                    i + 1, train_opt.init_num_epochs_S, acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                    acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                    acc.Mean_Intersection_over_Union()[1]))

            writer.add_scalar('NetLoss', NetLoss_aver, i + train_opt.init_num_epochs_G)
            writer.add_scalar('generator_loss', generator_loss_aver, i + train_opt.init_num_epochs_G)
            writer.add_scalar('l1_loss', l1_loss_aver, i + train_opt.init_num_epochs_G)
            writer.add_scalar('perception_loss', perception_loss_aver, i + train_opt.init_num_epochs_G)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i + train_opt.init_num_epochs_G)

            if train_opt.consist_flag == 'Y':
                print('Epochs: {}/{}, consist_loss: {:.4f}\n'.format(i + 1, train_opt.init_num_epochs_S, consist_loss_aver))
                f_log.write('Epochs: {}/{}, consist_loss: {:.4f}\n'.format(i + 1, train_opt.init_num_epochs_S, consist_loss_aver))
                writer.add_scalar('consist_loss', consist_loss_aver, i + train_opt.init_num_epochs_G)

            writer.add_scalar('Overall Accuracy:', acc.Pixel_Accuracy(), i + train_opt.init_num_epochs_G)
            writer.add_scalar('Precision Rate', acc.Pixel_Precision_Rate(), i + train_opt.init_num_epochs_G)
            writer.add_scalar('Recall Rate', acc.Pixel_Recall_Rate(), i + train_opt.init_num_epochs_G)
            writer.add_scalar('Kappa Coefficient:', acc.Pixel_Kappa(), i + train_opt.init_num_epochs_G)
            writer.add_scalar('F1', acc.Pixel_F1_score(), i + train_opt.init_num_epochs_G)
            writer.add_scalar('mIOU', acc.Mean_Intersection_over_Union()[0], i + train_opt.init_num_epochs_G)
            writer.add_scalar('cIOU', acc.Mean_Intersection_over_Union()[1], i + train_opt.init_num_epochs_G)

    
    # save reconstruct image
    save_gen_image(train_opt, dataset, netG_x2y, meanY, stdY, log_dir, 'complete', 'Seg')
    save_seg_image(train_opt, dataset, netG_x2y, netS, log_dir, 'complete', 'Seg')

    print('Start Training')
    f_log.write('Start Training\n')
    with torch.enable_grad():
        for i in range(train_opt.num_epochs):
            NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver, ssim_loss_aver = 0, 0, 0, 0, 0

            adjust_learning_rate(optimizerS, i, lr_start=1e-5, lr_max=1e-4)
            adjust_learning_rate(optimizerG_x2y, i, lr_start=1e-5, lr_max=1e-4)
                
            acc.reset()

            process_num = 0
            for data_array in train_dataloader:
                time_start = time.time()

                # Update G network:
                optimizerG_x2y.zero_grad()

                x = data_array[0]
                y = data_array[1]
                item = data_array[2]
                ref = data_array[3]

                x = x.to(device)
                y = y.to(device)
                y_fake = netG_x2y(x)
                cmap = netS(x, y)
                generator_loss, l1_loss, perception_loss, ssim_loss = criterion(y, y_fake, cmap)
                Loss = generator_loss + train_opt.perception_weight * perception_loss + train_opt.ssim_weight * ssim_loss

                Loss.backward(retain_graph=True)

                NetLoss = generator_loss + train_opt.l1_weight * l1_loss + train_opt.perception_weight * perception_loss + train_opt.ssim_weight * ssim_loss
                optimizerS.zero_grad()
                NetLoss.backward()

                optimizerG_x2y.step()
                optimizerS.step()

                NetLoss_aver += NetLoss * x.size(0) / len(dataset)
                generator_loss_aver += generator_loss * x.size(0) / len(dataset)
                l1_loss_aver += l1_loss * x.size(0) / len(dataset)
                perception_loss_aver += perception_loss * x.size(0) / len(dataset)
                ssim_loss_aver += ssim_loss * x.size(0) / len(dataset)

                cmask = torch.zeros_like(cmap)
                cmask[cmap > train_opt.prob_thresh] = 1
                for ns in range(x.size(0)):
                    change_mask = cmask[ns][0]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns][0].numpy()

                    item_x = math.floor(item[ns].numpy() / dataset.patch_count()[1])
                    item_y = item[ns].numpy() % dataset.patch_count()[1]
                    slice, _, _ = dataset.slice_assign(item_x, item_y)

                    acc.add_batch_map(ref_mask[pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16),
                                          change_mask[pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16),
                                          gt_map, pre_map)

                process_num += x.size()[0]
                time_end = time.time()
                time_per_iter = (time_end - time_start) / x.size()[0] * len(dataset)
                time_remaining = time_per_iter * (
                            (train_opt.num_epochs - 1 - i) + (1 - process_num / len(dataset)))
                time_desc_per = time_show(time_per_iter)
                time_desc = time_show(time_remaining)

                print('\rProcessing batch: {}/{}; Processing speed per iter: {}; Processing time remaining: {}'.format(
                        process_num, len(dataset), time_desc_per, time_desc), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Epochs: {}/{}, NetLoss: {:.4f}, generator_loss: {:.4f}, l1_loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}'.format(
                    i + 1, train_opt.num_epochs, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))

            print(
                'Epochs: {}/{}, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                    i + 1, train_opt.num_epochs, acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                    acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                    acc.Mean_Intersection_over_Union()[1]))

            f_log.write('Epochs: {}/{}, NetLoss: {:.4f}, generator_loss: {:.4f}, l1_loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}\n'.format(
                    i + 1, train_opt.num_epochs, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))

            f_log.write('Epochs: {}/{}, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}\n'.format(
                    i + 1, train_opt.num_epochs, acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                    acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                    acc.Mean_Intersection_over_Union()[1]))
            
            writer.add_scalar('NetLoss', NetLoss_aver, i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('generator_loss', generator_loss_aver, i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('l1_loss', l1_loss_aver, i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('perception_loss', perception_loss_aver, i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)

            writer.add_scalar('Overall Accuracy:', acc.Pixel_Accuracy(), i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('Precision Rate', acc.Pixel_Precision_Rate(), i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('Recall Rate', acc.Pixel_Recall_Rate(), i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('Kappa Coefficient:', acc.Pixel_Kappa(), i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('F1', acc.Pixel_F1_score(), i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('mIOU', acc.Mean_Intersection_over_Union()[0], i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)
            writer.add_scalar('cIOU', acc.Mean_Intersection_over_Union()[1], i + train_opt.init_num_epochs_G + train_opt.init_num_epochs_S)

    # save reconstruct image
    save_gen_image(train_opt, dataset, netG_x2y, meanY, stdY, log_dir, 'complete', 'Tra')
    save_seg_image(train_opt, dataset, netG_x2y, netS, log_dir, 'complete', 'Tra')
    
    writer.close()
    f_log.close()
    print('\r' + 'End of Saving', flush=True)

    path = os.path.join(log_dir, 'SModel_{}.pkl'.format(pattern_item))
    torch.save(netS.state_dict(), path)
    path = os.path.join(log_dir, 'GModel_{}.pkl'.format(pattern_item))
    torch.save(netG_x2y.state_dict(), path)  

    return netS, netG_x2y

def eval_result(train_opt, netS, netG, dataset, acc, log_dir, gt_map, pre_map, OutColorPath):   
    netS.eval()
    netG.eval()
    pad = dataset.overlap_padding

    test_dataloader = DataLoader(dataset, batch_size=train_opt.batch_size, shuffle=False)
    outDS = None
    with torch.no_grad():

        process_num = 0
        acc.reset()

        for data_array in test_dataloader:

            x = data_array[0]
            y = data_array[1]
            item = data_array[2]
            ref = data_array[3]

            x = x.to(device)
            y = y.to(device)
            cmap = netS(x, y)

            cmask = torch.zeros_like(cmap)
            cmask[cmap > train_opt.prob_thresh] = 1

            for ns in range(x.size(0)):
                write_cmap = cmap[ns].cpu().numpy()
                dataset.GDALwriteDefault(write_cmap, item[ns].numpy())

                # generate a color map indicating FP / FN / TP / TN
                if train_opt.write_color == True:
                    if outDS == None:
                        xsize, ysize, nband = dataset.size()
                        driver = dataset.imgDS_x.GetDriver()
                        outDS = driver.Create(OutColorPath, xsize, ysize, 1, gdal.GDT_Int32)
                        if outDS == None:
                            print("Cannot make a output raster")
                            sys.exit(0)

                        outDS.SetGeoTransform(dataset.imgDS_x.GetGeoTransform())
                        outDS.SetProjection(dataset.imgDS_x.GetProjection())

                    change_mask = cmask[ns]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns].numpy()
                    write_cmask = write_changemap_gdal(change_mask, ref_mask, write_color=train_opt.write_color, ref_map=gt_map, dt_map=pre_map)
                    dataset.GDALwrite(write_cmask.astype(np.int32), item[ns].numpy(), outDS)

                item_x = math.floor(item[ns].numpy() / dataset.patch_count()[1])
                item_y = item[ns].numpy() % dataset.patch_count()[1]
                slice, _, _ = dataset.slice_assign(item_x, item_y)

                acc.add_batch_map(ref_mask[0, pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16),
                                  change_mask[0, pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16),
                                  gt_map, pre_map)

            process_num += x.size()[0]
            print('\rProcessing batch: {}/{}'.format(process_num, len(dataset)), end='', flush=True)

        print('\r', end='', flush=True)

        print(
            'Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                acc.Mean_Intersection_over_Union()[1]))

    save_seg_image(train_opt, dataset, netG, netS, log_dir, 'complete', 'Tra')

    ParaTxtPath = os.path.join(log_dir,'Para_{}.txt'.format(time.strftime("%b%d%H%M%S", time.localtime())))
    TxtFile = open(ParaTxtPath, 'w')
    TxtFile.write("perception_weight:{}\n".format(train_opt.perception_weight))
    TxtFile.write("ssim_weight:{}\n".format(train_opt.ssim_weight))
    TxtFile.write("perception_perBand:{}\n".format(train_opt.perception_perBand))
    TxtFile.write("perception_layer:{}\n".format(train_opt.perception_layer))
    TxtFile.write("l1_weight:{}\n".format(train_opt.l1_weight))
    TxtFile.write("discriminator_continuous:{}\n".format(train_opt.discriminator_continuous))
    TxtFile.write("prob_thresh:{}\n".format(train_opt.prob_thresh))
    TxtFile.write(
        "Segmentation, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIOU:{:.4f}\n".format(
            acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
            acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
            acc.Mean_Intersection_over_Union()[1]))
    TxtFile.write("tips:eval_patch\n")

    TxtFile.close()

if __name__ == '__main__':
    main()