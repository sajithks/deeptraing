# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:27:04 2015

@author: root
"""

#%%



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
#import time
import scipy.io as sio

from scipy.ndimage import label

def fthreshValue(fmeasure,binval):
    a = np.linspace(0, 1, binval)
    fthresh = np.zeros((a.shape[0]))
    for ii in range(a.shape[0]):
        fthresh[ii] = (fmeasure > a[ii] ).tolist().count(True)
    return(fthresh/fmeasure.shape[0])

bincount = 100
LINEWIDTH = 1
FONTSIZE = 16
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : FONTSIZE}

plt.rc('font', **font)

#%%



gtfolder = '/Users/sajithks/Documents/dump/data_neutrophils/ilastik/'

rffolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/randomforest/ver3/'
neuralfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/neuralnet_caffedirect/ver3/'
#% gt
startime = time.time()
gt = cv2.imread(gtfolder+'Labels00.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)

mask = gt==1
seed, ncc = label(gt==2,np.ones((3,3)))
gtlabel = cb.watershedSeeded(np.ones(gt.shape), mask, seed)
#gtlabel[gtlabel==-1]=0
#gtlabel[gtlabel==1]=0




rfseg = cv2.imread(rffolder+'neutro_conv_1_2_4_Aligned0000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
rfseg = rfseg[6:rfseg.shape[0]-6, 6:rfseg.shape[1]-6, :]

segimg = np.argmax(rfseg, 2)
segimglab, segncc = label( segimg==0, np.ones((3,3)) )
segseed = (segimg==2)*1
segseed = segseed*(sp.ndimage.binary_fill_holes(segimg==0)>0)

#for ii in np.arange(1, segncc,1):
#    if( (np.sum((1*(segimglab == ii) )*segseed) == 0) ):
#        segimglab[segimglab==ii] = 0
#        

segimglab = cb.removeBoundaryLabel(segimglab)
#segseed = segseed*segimglab>0
    
segmask = (segimglab>1)*1

segseed, ncc = label(segseed, np.ones((3,3)))
#segseed = cb.removeBoundaryLabel(segseed)

segimglab = cb.watershedSeeded(np.ones(segmask.shape), segmask, segseed)

#segimglab[segimglab==-1]=0
#segimglab[segimglab==1]=0

rowdiff = gt.shape[0]-segimg.shape[0]
coldiff = gt.shape[1]-segimg.shape[1]
gtlabel = gtlabel[rowdiff/2:(gt.shape[0]-rowdiff/2), coldiff/2:(gt.shape[1]-coldiff/2)]
#gtlab, nccgt = label(gt,np.ones((3,3)))

gtlabel = cb.labelDilation(gtlabel,7)

#%%
recall, precision, fmeasure, dicescore = cb.evaluateSegmentation2(gtlabel, segimglab)
#print recall[1],precision[1], fmeasure[1]
bincount = 100
ft1 = fthreshValue(fmeasure, bincount)
xval = np.linspace(0, 1, bincount)


plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
line1, = plt.plot(xval, ft1, label='CBA',color = 'k',linewidth = LINEWIDTH)






#%%
a1 = cb.findAreaDistribution(gtlabel)
a2 = cb.findAreaDistribution(segimglab)
#a3 = cb.findAreaDistribution(cbafilt)
#a4 = cb.findAreaDistribution(cbasmooth)
##a5 = cb.findAreaDistribution(microb)


bincount=50
ah1, bina1 = np.histogram(a1,bincount)
ah2, bina2 = np.histogram(a2,bincount)

LINEWIDTH = 2
plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
line1, = plt.plot( bina1[:-1], np.float32(ah1)/ah1.sum(), label='ground truth',color = 'c',linewidth = LINEWIDTH)
line2, = plt.plot(bina2[:-1],np.float32(ah2)/ah2.sum(), label='CBA',color = 'k',linewidth = LINEWIDTH)
