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
import glob
#import time
import scipy.io as sio
import criticalpoints2 as cp
import string
from scipy.ndimage import label

def fthreshValue(fmeasure,binval):
    a = np.linspace(0, 1, binval)
    fthresh = np.zeros((a.shape[0]))
    for ii in range(a.shape[0]):
        fthresh[ii] = (fmeasure > a[ii] ).tolist().count(True)
    return(fthresh/fmeasure.shape[0])

def labelgt(gt):
    mask = gt==1
    seed, ncc = label(gt==2,np.ones((3,3)))
    gtlabel = cb.watershedSeeded(np.ones(gt.shape), mask, seed)
    return(gtlabel)

def labelSegimg(img):
    segimg = np.argmax(img, 2)
    segimglab, segncc = label( segimg==0, np.ones((3,3)) )
    segseed = (segimg==2)*1
    segseed = segseed*(sp.ndimage.binary_fill_holes(segimg==0)>0)
    
    segimglab = cb.removeBoundaryLabel(segimglab)        
    segmask = (segimglab>1)*1    
    segseed, ncc = label(segseed, np.ones((3,3)))
    #segseed = cb.removeBoundaryLabel(segseed)
    
    segimglab = cb.watershedSeeded(np.ones(segmask.shape), segmask, segseed)
    
    return(segimglab)

bincount = 100
LINEWIDTH = 1
FONTSIZE = 16
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : FONTSIZE}

plt.rc('font', **font)

#%%



gtfolder = '/Users/sajithks/Documents/dump/data_neutrophils/ilastik/'

#neural net
neuralfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/neuralnet_caffedirect/ver3/'
neuraltrainfiles = sorted(glob.glob(neuralfolder + '*Aligned0000.tif'))
neuraltestfiles = sorted(glob.glob(neuralfolder + '*Aligned0010.tif'))

#random forest
rffolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/randomforest/ver3/'
rftrainfiles = sorted(glob.glob(rffolder + '*Aligned0000.tif'))
rftestfiles = sorted(glob.glob(rffolder + '*Aligned0010.tif'))

#% ground truth
startime = time.time()
gttrain = cv2.imread(gtfolder+'Labels00.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
gttest = cv2.imread(gtfolder+'Labels10.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)


gttrainlab = labelgt(gttrain)
gttestlab = labelgt(gttest)

# training set performance
rftrain = []
ntrain = []
# 
for ii in range(np.shape(rftrainfiles)[0]):
    rfseg = cv2.imread(rftrainfiles[ii], cv2.CV_LOAD_IMAGE_UNCHANGED)
    rfseg = rfseg[32:rfseg.shape[0]-32, 30:rfseg.shape[1]-30, :]
    
    # neural resize
    neuralseg = cv2.imread(neuraltrainfiles[ii], cv2.CV_LOAD_IMAGE_UNCHANGED)
    neuralseg = neuralseg[40:neuralseg.shape[0]-40, 40:neuralseg.shape[1]-40, :]
    
    rfseglab = labelSegimg(rfseg)
    neuralseglab = labelSegimg(neuralseg)
    
    rowdiff = gttrainlab.shape[0]-rfseg.shape[0]
    coldiff = gttrainlab.shape[1]-rfseg.shape[1]
    gttrainlab = gttrainlab[rowdiff/2:(gttrainlab.shape[0]-rowdiff/2), coldiff/2:(gttrainlab.shape[1]-coldiff/2)]
    #gtlab, nccgt = label(gt,np.ones((3,3)))
    
    gttrainlab = cb.labelDilation(gttrainlab,2)
    
    #%%
    recall1, precision1, fmeasure1, dicescore1 = cb.evaluateSegmentation2(gttrainlab, rfseglab)
    recall2, precision2, fmeasure2, dicescore2 = cb.evaluateSegmentation2(gttrainlab, neuralseglab)

    rftrain.append(fmeasure1.sum())
    ntrain.append(fmeasure2.sum())
    
    #print recall[1],precision[1], fmeasure[1]
    bincount = 100
    ftrf = fthreshValue(fmeasure1, bincount)
    ftneural = fthreshValue(fmeasure2, bincount)
    xval = np.linspace(0, 1, bincount)
    
    
#    plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
#    line1, = plt.plot(xval, ftrf, label='randomforest',color = 'k',linewidth = LINEWIDTH)
#    line2, = plt.plot(xval, ftneural, label='neural',color = 'b',linewidth = LINEWIDTH)


#%% testset performance

rftest = []
ntest = []
 
for ii in range(np.shape(rftestfiles)[0]):
    rfseg = cv2.imread(rftestfiles[ii], cv2.CV_LOAD_IMAGE_UNCHANGED)
    rfseg = rfseg[32:rfseg.shape[0]-32, 30:rfseg.shape[1]-30, :]
    
    # neural resize
    neuralseg = cv2.imread(neuraltestfiles[ii], cv2.CV_LOAD_IMAGE_UNCHANGED)
    neuralseg = neuralseg[40:neuralseg.shape[0]-40, 40:neuralseg.shape[1]-40, :]
    
    rfseglab = labelSegimg(rfseg)
    neuralseglab = labelSegimg(neuralseg)
    
    rowdiff = gttestlab.shape[0]-rfseg.shape[0]
    coldiff = gttestlab.shape[1]-rfseg.shape[1]
    gttestlab = gttestlab[rowdiff/2:(gttestlab.shape[0]-rowdiff/2), coldiff/2:(gttestlab.shape[1]-coldiff/2)]
    #gtlab, nccgt = label(gt,np.ones((3,3)))
    
    gttrainlab = cb.labelDilation(gttrainlab, 2)
    
    #%
    recall1, precision1, fmeasure1, dicescore1 = cb.evaluateSegmentation2(gttestlab, rfseglab)
    recall2, precision2, fmeasure2, dicescore2 = cb.evaluateSegmentation2(gttestlab, neuralseglab)
    
    rftest.append(fmeasure1.sum())
    ntest.append(fmeasure2.sum())    
    
    #print recall[1],precision[1], fmeasure[1]
    bincount = 100
    ftrf = fthreshValue(fmeasure1, bincount)
    ftneural = fthreshValue(fmeasure2, bincount)
    xval = np.linspace(0, 1, bincount)
    
    
    plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
    line1, = plt.plot(xval, ftrf, label='randomforest',color = 'k',linewidth = LINEWIDTH)
    line2, = plt.plot(xval, ftneural, label='neural',color = 'b',linewidth = LINEWIDTH)

#%%
rftrain = np.array(rftrain)
ntrain = np.array(ntrain)

rftest = np.array(rftest)
ntest = np.array(ntest)

netnames = []
for ii in neuraltrainfiles:    
    netnames.append(string.split(string.split(ii,'/')[-1], '_iter')[0])

#%%
plt.figure()
line1, = plt.plot(range(rftrain.shape[0]), rftrain, label='rf', color='k')
line2, = plt.plot(range(ntrain.shape[0]), ntrain, label='neural', color = 'b')
plt.xlabel('network model ', fontsize=20),plt.ylabel('Area under cumulative fscore', fontsize=20)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=40)},loc='lower left')
#plt.show()
frame=plt.gca()
#plt.yticks(np.arange(0, 1.25, .25),np.arange(0, 125, 25))
plt.xticks(range(len(netnames)),netnames)
plt.tick_params(axis='both', which='major', labelsize=15)

#%%
plt.figure()
line1, = plt.plot(range(rftest.shape[0]), rftest, label='rf', color='k')
line2, = plt.plot(range(ntest.shape[0]), ntest,label='neural', color = 'b')
plt.xlabel('network model ', fontsize=20),plt.ylabel('Area under cumulative fscore', fontsize=20)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=40)},loc='lower left')
#plt.show()
frame=plt.gca()
#plt.yticks(np.arange(0, 1.25, .25),np.arange(0, 125, 25))
plt.xticks(range(len(netnames)),netnames)
plt.tick_params(axis='both', which='major', labelsize=15)




#%%

#ma,mi,sa,mva=cp.findCriticalPoints(rfseg[:,:,2],10.)
#
#simg = np.zeros_like(rfseg[:,:,2])
#for ii in ma:
#    simg[ii[0],ii[1]] = 1
#cb.myshow(simg)

#a1 = cb.findAreaDistribution(gttrainlab)
#a2 = cb.findAreaDistribution(rfseglab)
#a3 = cb.findAreaDistribution(neuralseglab)
##a4 = cb.findAreaDistribution(cbasmooth)
###a5 = cb.findAreaDistribution(microb)
#
##
#bincount=50
#ah1, bina1 = np.histogram(a1,bincount)
#ah2, bina2 = np.histogram(a2,bincount)
#ah3, bina3 = np.histogram(a3,bincount)
#
##
#LINEWIDTH = 2
#plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
#line1, = plt.plot( bina1[:-1], np.float32(ah1)/ah1.sum(), label='ground truth',color = 'c',linewidth = LINEWIDTH)
#line2, = plt.plot(bina2[:-1],np.float32(ah2)/ah2.sum(), label='rf',color = 'k',linewidth = LINEWIDTH)
#line3, = plt.plot(bina3[:-1],np.float32(ah3)/ah3.sum(), label='neural',color = 'b',linewidth = LINEWIDTH)
