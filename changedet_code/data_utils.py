import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import os
import sys
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np
import cv2
from tqdm import tqdm
import gc
import math
import random
import pdb

from PIL import Image
import numpy as np

# ERROR 1: PROJ
os.environ['PROJ_LIB'] = r'/data/chen.wu/anaconda3/share/proj/'

# dataset to read remote sensing images with gdal
# the read patch is obtained from the large-scale image with overlaps
# when writing the patches, only the centering region without overlap padding is written
class GDALDataset(Dataset):

    def __init__(self, imgPathX, imgPathY, refPath=None, outPath=None, transforms=None, \
                enhance=None, patch_size=(200, 200), overlap_padding=(10, 10), \
                aug_flag=None, stage=None, consist_flag=None):
        super(GDALDataset, self).__init__()
        self.imgPathX = imgPathX
        self.imgDS_x = gdal.Open(imgPathX)

        if self.imgDS_x is None:
            print('No such a Image file:{}'.format(imgPathX))
            sys.exit(0)
        xsize = self.imgDS_x.RasterXSize
        ysize = self.imgDS_x.RasterYSize
        nband = self.imgDS_x.RasterCount

        self.imgPathY = imgPathY
        self.imgDS_y = gdal.Open(imgPathY)

        if self.imgDS_y is None:
            print('No such a Image file:{}'.format(imgPathY))
            sys.exit(0)
        xsize2 = self.imgDS_y.RasterXSize
        ysize2 = self.imgDS_y.RasterYSize
        nband2 = self.imgDS_y.RasterCount

        if xsize != xsize2 or ysize != ysize2 or nband != nband2:
            print('Image sizes don\'t match')
            sys.exit(0)

        self.transforms = transforms
        self.enhance = enhance

        xstart = list(range(0, xsize, patch_size[0] - 2 * overlap_padding[0]))
        xend = [(x + patch_size[0]  - 2 * overlap_padding[0]) for x in xstart if (x + patch_size[0] - 2 * overlap_padding[0] < xsize)]
        xend.append(xsize)

        ystart = list(range(0, ysize, patch_size[1] - 2 * overlap_padding[1]))
        yend = [(y + patch_size[1] - 2 * overlap_padding[1]) for y in ystart if (y + patch_size[1] - 2 * overlap_padding[1] < ysize)]
        yend.append(ysize)

        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend

        self.patch_size = patch_size
        self.overlap_padding = overlap_padding

        self.refPath = refPath
        if refPath is not None:
            self.imgDS_ref = gdal.Open(refPath)
            if self.imgDS_ref is None:
                print('No such a Image file:{}'.format(refPath))
                sys.exit(0)
            xsize3 = self.imgDS_ref.RasterXSize
            ysize3 = self.imgDS_ref.RasterYSize
            nband3 = self.imgDS_ref.RasterCount
            if xsize != xsize3 or ysize != ysize3 or nband3 != 1:
                print('Reference sizes don\'t match image')
                sys.exit(0)
        else:
            self.imgDS_ref = None

        self.outPath = outPath
        self.outDS = None
        self.aug_flag = aug_flag
        self.stage = stage
        self.consist_flag = consist_flag


    def __getitem__(self, item):
        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        xsize, ysize, nband = self.size()

        tmp_x = []
        tmp_y = []
        for b in range(nband):
            tmp_x.append(self.imgDS_x.GetRasterBand(b + 1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2], slice_read[3]))
            tmp_y.append(self.imgDS_y.GetRasterBand(b + 1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2], slice_read[3]))

        tmp_x = np.array(tmp_x, dtype=float)
        tmp_y = np.array(tmp_y, dtype=float)

        if self.enhance is not None:
            tmp_x = self.enhance(tmp_x, switch=1)
            tmp_y = self.enhance(tmp_y, switch=2)
        
        msImage_x = np.zeros((nband, self.patch_size[1], self.patch_size[0]), dtype=float)
        msImage_y = np.zeros((nband, self.patch_size[1], self.patch_size[0]), dtype=float)

        msImage_x[:, slice_write[1]:slice_write[1] + slice_write[3],
        slice_write[0]:slice_write[0] + slice_write[2]] = tmp_x
        msImage_y[:, slice_write[1]:slice_write[1] + slice_write[3],
        slice_write[0]:slice_write[0] + slice_write[2]] = tmp_y

        msImage_x = torch.from_numpy(msImage_x).float()
        msImage_y = torch.from_numpy(msImage_y).float()
        item = torch.tensor(item)

        if self.transforms is not None:
            msImage_x, sync = self.transforms(msImage_x)
            msImage_y, sync = self.transforms(msImage_y, sync)

        refImage = np.zeros((1, self.patch_size[1], self.patch_size[0]), dtype=float)
        if self.imgDS_ref is not None:
            tmp_ref = []
            tmp_ref.append(self.imgDS_ref.GetRasterBand(1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2],
                                                                       slice_read[3]))
            tmp_ref = np.array(tmp_ref, dtype=float)
            refImage[:, slice_write[1]:slice_write[1] + slice_write[3],
            slice_write[0]:slice_write[0] + slice_write[2]] = tmp_ref
        refImage = torch.from_numpy(refImage).float()

        if self.aug_flag == 'Y' and self.stage == 'train':
            augment_flag_list = [0, 1, 2, 3]
            select_flag = random.choice(augment_flag_list)

            # original image
            if self.consist_flag == 'Y':
                if select_flag != 0:
                    select_ori_flag = 0
                    msImage_x_ori, msImage_y_ori, refImage_ori = msImage_x, msImage_y, refImage
                else:
                    augment_ori_flag_list = [1, 2, 3]
                    select_ori_flag = random.choice(augment_ori_flag_list)
                    msImage_x_ori = self.augmentation_image(msImage_x, select_ori_flag)
                    msImage_y_ori = self.augmentation_image(msImage_y, select_ori_flag)
                    refImage_ori = self.augmentation_image(refImage, select_ori_flag)

                msImage_x = self.augmentation_image(msImage_x, select_flag)
                msImage_y = self.augmentation_image(msImage_y, select_flag)
                refImage = self.augmentation_image(refImage, select_flag)

                return msImage_x, msImage_y, item, refImage, msImage_x_ori, msImage_y_ori, refImage_ori, select_ori_flag, select_flag
            else:
                msImage_x = self.augmentation_image(msImage_x, select_flag)
                msImage_y = self.augmentation_image(msImage_y, select_flag)
                refImage = self.augmentation_image(refImage, select_flag)
                return msImage_x, msImage_y, item, refImage

        return msImage_x, msImage_y, item, refImage

    def easy_see(self, file, file_name):
        from PIL import Image
        file_path = file_name + '.png'
        if len(file) == 1:
            file = ((file[0,:,:].numpy()+1.5)*63).astype(np.uint8)
            file_img = Image.fromarray(file, mode='L')
            file_img.save(file_path)
        else:
            file = (((file[:3,:,:].numpy()+1.5).transpose((1,2,0)))*63).astype(np.uint8)
            file_img = Image.fromarray(file)
            file_img.save(file_path)

    def augmentation_image(self, patch_item, select_flag):
        # original
        if select_flag == 0:
            return patch_item
        # horizontal
        elif select_flag == 1:
            return torch.flip(patch_item, dims=[1])
        # vertical
        elif select_flag == 2:
            return torch.flip(patch_item, dims=[2])
        # transpose
        else:
            return torch.transpose(patch_item, 1, 2)

    def __len__(self):
        return len(self.xstart) * len(self.ystart)

    def patch_count(self):
        return len(self.xstart), len(self.ystart)

    def size(self):
        xsize = self.imgDS_x.RasterXSize
        ysize = self.imgDS_x.RasterYSize
        nband = self.imgDS_x.RasterCount
        return xsize, ysize, nband

    def slice_assign(self, item_x, item_y):

        pad = self.overlap_padding
        xsize, ysize, nband = self.size()

        xstart = self.xstart[item_x]
        xend = self.xend[item_x]
        ystart = self.ystart[item_y]
        yend = self.yend[item_y]
        slice = (xstart, ystart, xend - xstart, yend - ystart)

        x_ori = 0 if xstart - pad[0] > 0 else pad[0]
        y_ori = 0 if ystart - pad[1] > 0 else pad[1]

        xstart = xstart - pad[0] if xstart - pad[0] > 0 else 0
        ystart = ystart - pad[1] if ystart - pad[1] > 0 else 0
        xend = xend + pad[0] if xend + pad[0] < xsize else xsize
        yend = yend + pad[1] if yend + pad[1] < ysize else ysize
        slice_read = (xstart, ystart, xend - xstart, yend - ystart)

        slice_write = (x_ori, y_ori, xend - xstart, yend - ystart)

        return slice, slice_read, slice_write

    def GDALwriteDefault(self, outImage, item):
        # Only write one-band image

        if self.outPath == None:
            dir, fname = os.path.split(self.imgPathX)
            fname, ext = os.path.splitext(fname)
            fname = "{}_cmp{}".format(fname, ext)
            outPath = os.path.join(dir, fname)
            self.outPath = outPath

        xsize, ysize, nband = self.size()

        if self.outDS == None:
            driver = self.imgDS_x.GetDriver()
            self.outDS = driver.Create(self.outPath, xsize, ysize, 1, gdal.GDT_Float32)
            if self.outDS == None:
                print("Cannot make a output raster")
                sys.exit(0)

            self.outDS.SetGeoTransform(self.imgDS_x.GetGeoTransform())
            self.outDS.SetProjection(self.imgDS_x.GetProjection())

            outBand = self.outDS.GetRasterBand(1)
            # outBand.SetNoDataValue(0)
        else:
            outBand = self.outDS.GetRasterBand(1)

        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        pad = self.overlap_padding
        outBand.WriteArray(outImage[0, pad[1]:pad[1]+slice[3], pad[0]:pad[0]+slice[2]], slice[0], slice[1])

    def GDALwrite(self, outImage, item, outGDAL=None):

        if outGDAL == None:
            self.GDALwriteDefault(outImage.numpy(), item)
            return

        if outImage.shape[0] != outGDAL.RasterCount:
            print('The band of output image doesn\'t match the output GDAL dataset')
            sys.exit(0)

        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        pad = self.overlap_padding

        for b in range(outGDAL.RasterCount):
            outBand = outGDAL.GetRasterBand(b + 1)
            outBand.WriteArray(outImage[b, pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]], slice[0], slice[1])