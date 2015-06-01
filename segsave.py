# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:05:03 2015

@author: root
"""
import time
import sys
import os
sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf')
sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/ecoli_det_seg')

import createbacteria  as cb
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import cv2
import glob
#import time
import scipy.io as sio
import criticalpoints2 as cp
import string
from scipy.ndimage import label

#%%
orimgfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/sampimg/'

neuralfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/neuralnet_caffedirect/ver4/'
nntestfiles = sorted(glob.glob(neuralfolder + '*.png'))

gtfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/labels/fulllabels/'




outfolder1 = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/neuralnet_caffedirect/seg/'
outfolder2 = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/randomforest/seg/'
outfolder3 = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/overlay/'



for ii in nntestfiles:
    imgname = string.split(string.split(string.split(ii,'/')[-1], 'Aligned')[1],'.' )[0]
    orimg = cv2.imread(sorted(glob.glob(orimgfolder + '*'+imgname +'*'))[0], -1 )

    savname = string.split(ii,'/')[-1]
    img = cv2.imread(ii,-1)
    outimg = np.argmax(img,2)
    outimgnew = ((outimg==0)+(outimg==2) )*1  
    
    outimgnew = outimgnew[40:outimgnew.shape[0]-40, 40+104:outimgnew.shape[1]-40-104]
    
    rowdiff = orimg.shape[0]-outimgnew.shape[0]
    coldiff = orimg.shape[1]-outimgnew.shape[1]
    orimg = orimg[rowdiff/2:(orimg.shape[0]-rowdiff/2), coldiff/2:(orimg.shape[1]-coldiff/2)]

    cv2.imwrite(outfolder1+savname, cb.overlayImage(orimg, outimgnew))

#random forest
rffolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/randomforest/ver4/'
rftrainfiles = sorted(glob.glob(rffolder + '*.png'))

for ii in rftrainfiles:
    imgname = string.split(string.split(string.split(ii,'/')[-1], 'Aligned')[1],'.' )[0]
    orimg = cv2.imread(sorted(glob.glob(orimgfolder + '*'+imgname +'*'))[0], -1 )
    
    savname = string.split(ii,'/')[-1]
    img = cv2.imread(ii,-1)
    outimg = np.argmax(img,2)
    outimgnew = ((outimg==0)+(outimg==2) )*1  
    outimgnew = outimgnew[32:outimgnew.shape[0]-32, 30+104:outimgnew.shape[1]-30-104]
    
    rowdiff = orimg.shape[0]-outimgnew.shape[0]
    coldiff = orimg.shape[1]-outimgnew.shape[1]
    orimg = orimg[rowdiff/2:(orimg.shape[0]-rowdiff/2), coldiff/2:(orimg.shape[1]-coldiff/2)]
        
    
    cv2.imwrite(outfolder2+savname, cb.overlayImage(orimg, outimgnew))



#%%
gtfiles = sorted(glob.glob(gtfolder + '*.png'))

for ii in gtfiles:
    imgname = string.split(string.split(string.split(ii,'/')[-1], 'Aligned')[1],'.' )[0]
#    print imgname
    orimg = cv2.imread(sorted(glob.glob(orimgfolder + '*'+imgname +'*'))[0], -1 )
    gtimg = cv2.imread(sorted(glob.glob(gtfolder + '*'+imgname +'*'))[0], -1 )
    savname = string.split(ii,'/')[-1]

    rowdiff = gtimg.shape[0]-outimgnew.shape[0]
    coldiff = gtimg.shape[1]-outimgnew.shape[1] 

    gtimg = gtimg[rowdiff/2:(gtimg.shape[0]-rowdiff/2), coldiff/2:(gtimg.shape[1]-coldiff/2)]
    orimg = orimg[rowdiff/2:(orimg.shape[0]-rowdiff/2), coldiff/2:(orimg.shape[1]-coldiff/2)]

    gtimg = ((gtimg==1)+(gtimg==2) )*1  

    cv2.imwrite(outfolder3+savname, cb.overlayImage(orimg, gtimg))











