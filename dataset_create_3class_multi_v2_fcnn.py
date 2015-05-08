# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:35:28 2015

@author: Sajith
"""



import scipy as sp
import time
#import sys
import os
#sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf')
#sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/ecoli_det_seg')
#import createbacteria  as cb
import numpy as np
#from matplotlib import pyplot as plt
#import skimage.morphology as skmorph
import cv2
from scipy.ndimage import label
from scipy.ndimage import interpolation
from random import shuffle
import glob
import subprocess
import signal
import Image
caffe_root = '/home/saj/Downloads/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import time
import sklearn
import random
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as ab
from sklearn import svm
from multiprocessing import Pool

print 'libraries loaded'

def fastConvolve(inimg, filt):
    outnums = np.shape(filt)[0]
    
    convrow = inimg.shape[1] - 2*(filt.shape[2]/2)
    convcol = inimg.shape[2] - 2*(filt.shape[3]/2)
    
    convout = np.zeros((outnums, convrow, convcol))
    for ii in range(outnums):
        filterval = filt[ii]
        convout[ii,:,:] = sp.signal.convolve(inimg, filterval,'valid')[0]
    return convout

def checkEven(inimg):
    if(np.mod(inimg.shape[1],2 )==1 ):
        inimg = inimg[:,0:inimg.shape[1]-1,:]
    if(np.mod(inimg.shape[2], 2 )==1 ):
        inimg = inimg[:,:,0:inimg.shape[2]-1]
    return(inimg)

def fastMaxpool(inimg):
    inimg = checkEven(inimg)
    outimg = np.zeros_like(inimg)
    for rowoffset in range(2):
        for coloffset in range(2):
            tempimg = inimg[:,rowoffset:inimg.shape[1]:1,coloffset:inimg.shape[2]:1]
            poolmax = []
            for poffsetrow in range(2):
                for poffsetcol in range(2):
                    poolmax.append(tempimg[:,poffsetrow:tempimg.shape[1]-rowoffset:2, poffsetcol:tempimg.shape[2]-coloffset:2])
            subpool = np.max(poolmax,0)
            
            outimg[:,rowoffset:outimg.shape[1]-rowoffset:2, coloffset:outimg.shape[2]-coloffset:2] = subpool

    return(outimg)

#%% 
#print 'network training done'
###############################################################################


caffe.set_mode_cpu()
#net = caffe.Net(caffe_root + 'examples/ecoli/ecolifile2deploy.prototxt',
#                caffe_root + 'examples/ecoli/file8bit2_iter_10000.caffemodel',
#                caffe.TEST)
net = caffe.Net(caffe_root + 'examples/ecoli/neutro3classv3_deploy.prototxt',
                caffe_root + 'examples/ecoli/neutro3clasv3_iter_10000.caffemodel',
                caffe.TEST)

######################## training #######################################
#%% read images
print 'loading image and label ...'
orimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/img/Aligned0000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0003.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
orimg = np.float32(orimg)
orimg = orimg-orimg.min()
orimg = np.uint8(255*(orimg/orimg.max()))

#segimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/seg/labeled20141021_ex_Phase0003.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
segimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/seg/labeledAligned0000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)


labimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/seg/Aligned00003class.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)

print('image and label loaded')
#%%
print 'extracting features ...'
#st = time.time()

#pval = ['conv3','conv3_level2','conv3_level3']

#level 1
inimg = np.zeros((3,orimg.shape[0],orimg.shape[1]))
inimg[0,:,:] = orimg
inimg[1,:,:] = orimg
inimg[2,:,:] = orimg

filtname = 'conv3'
filt = net.params[filtname][0].data
maxoutl1 = fastMaxpool(fastConvolve(inimg, filt))

#%level 2

filtname = 'conv3_level2'
filt = net.params[filtname][0].data
maxoutl2 = []
coutl2 = []
for ii in range(2):
    for jj in range(2):
        maxoutl2.append(fastMaxpool(fastConvolve(maxoutl1[:, ii:maxoutl1.shape[1]:2, jj:maxoutl1.shape[2]:2], filt)))

#level 3

filtname = 'conv3_level3'
filt = net.params[filtname][0].data

maxoutl3 = []
coutl3 = []
for kk in range(np.shape(maxoutl2)[0] ):
    for ii in range(2):
        for jj in range(2):
            maxoutl3.append( fastMaxpool(fastConvolve(maxoutl2[kk][:, ii:np.shape(maxoutl2)[2]:2, jj:np.shape(maxoutl2)[3]:2 ], filt)))

#roll out to full size
dim1 = np.shape(maxoutl3)[1]
dim2 = np.int32(np.sqrt((np.shape(maxoutl3)[0])) * np.shape(maxoutl3)[2])
dim3 = np.int32(np.sqrt(np.shape(maxoutl3)[0]) * np.shape(maxoutl3)[3])
featimg = np.zeros((dim1, dim2, dim3 ))

for ii in range(4):
    for jj in range(4):
        featimg[:, ii:dim2:4, jj:dim3:4] = maxoutl3[ii*2+jj*1]
        
print 'features extracted'

#%%
rowstart = orimg.shape[0]-featimg.shape[1]+1
rowend = featimg.shape[1]-1
colstart = orimg.shape[1]-featimg.shape[2]+1
colend = featimg.shape[2]-1

print 'training ...'
feat = []
lab = []

for ii in range(10000):
    row = random.randint(rowstart, rowend)
    col = random.randint(colstart, colend)
    maxpoolrow = row - (orimg.shape[0]-featimg.shape[1])/2
    maxpoolcol = col - (orimg.shape[1]-featimg.shape[2])/2
    
    feat.append(featimg[:, maxpoolrow, maxpoolcol])
    lab.append(labimg[row,col])
#for row in np.int32(np.linspace(0, orimg.shape[0]-1, 100)):
#    for col in np.int32(np.linspace(0, orimg.shape[1]-1, 100)):
#        feat.append(featmat[:, row, col])
#        lab.append(labimg[row, col])
#    
feat = np.array(feat)
lab = np.array(lab)
#
rforest = rf(n_estimators=200)
#
rforest.fit(feat,lab)
print 'training done.'
#%%

#
##%%####################################################################
################################ testing ###############################
########################################################################

print 'test data loading ...'

imgfolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/sampimg/'
outfolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/output/neuralnet/'
inputimgfiles = sorted(glob.glob(imgfolder + '*.tif'))
#st = time.time()
count = 0
for infile in inputimgfiles:
    
    orimg = cv2.imread(infile, cv2.CV_LOAD_IMAGE_UNCHANGED)
    #orimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/img/Aligned0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
    #orimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0004.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
    orimg = np.float32(orimg)
    orimg = orimg-orimg.min()
    orimg = np.uint8(255*(orimg/orimg.max()))
    #
    print 'extracting features ...'
    #st = time.time()
    
    #pval = ['conv3','conv3_level2','conv3_level3']
    
    #level 1
    inimg = np.zeros((3,orimg.shape[0],orimg.shape[1]))
    inimg[0,:,:] = orimg
    inimg[1,:,:] = orimg
    inimg[2,:,:] = orimg
    
    filtname = 'conv3'
    filt = net.params[filtname][0].data
    maxoutl1 = fastMaxpool(fastConvolve(inimg, filt))
    
    #%level 2
    
    filtname = 'conv3_level2'
    filt = net.params[filtname][0].data
    maxoutl2 = []
    coutl2 = []
    for ii in range(2):
        for jj in range(2):
            maxoutl2.append(fastMaxpool(fastConvolve(maxoutl1[:, ii:maxoutl1.shape[1]:2, jj:maxoutl1.shape[2]:2], filt)))
    
    #level 3
    
    filtname = 'conv3_level3'
    filt = net.params[filtname][0].data
    
    maxoutl3 = []
    coutl3 = []
    for kk in range(np.shape(maxoutl2)[0] ):
        for ii in range(2):
            for jj in range(2):
                maxoutl3.append( fastMaxpool(fastConvolve(maxoutl2[kk][:, ii:np.shape(maxoutl2)[2]:2, jj:np.shape(maxoutl2)[3]:2 ], filt)))
    
    #roll out to full size
    dim1 = np.shape(maxoutl3)[1]
    dim2 = np.int32(np.sqrt((np.shape(maxoutl3)[0])) * np.shape(maxoutl3)[2])
    dim3 = np.int32(np.sqrt(np.shape(maxoutl3)[0]) * np.shape(maxoutl3)[3])
    featimg = np.zeros((dim1, dim2, dim3 ))
    
    for ii in range(4):
        for jj in range(4):
            featimg[:, ii:dim2:4, jj:dim3:4] = maxoutl3[ii*2+jj*1]
            
    print 'features extracted'
    #%
    st =time.time()
#    outimg = np.zeros((featimg.shape[1], featimg.shape[2], 3),dtype= np.float32)
    
#    for ii in range(featimg.shape[2]):
#        print ii
    #    outimg[ :,ii] = rforest.predict(featimg[:,:,ii].T)
    #    outimg[ii,:] = adaboost.predict_proba(featmat[:,ii,:].T)[:,0]
#        outimg[ :,ii,:] = rforest.predict_proba(featimg[:,:,ii].T)
    #    outimg[ii,:] = svmclas.predict_proba(featmat[:,ii,:].T)[:,0]
    
      
#    print 'testing time ', time.time() - st
    
#    plt.imshow(outimg)
    
    
#%

#for ii in range(np.shape(filt)[0]):
#    for jj in range(np.shape(filt)[1]):
#        plt.imshow(filt[ii][jj],cmap='gray'),plt.show()

#%########################################

#fully connected layer

    filtname = 'ip1'
    ipfilt = net.params[filtname][0].data
    outimg = np.zeros((featimg.shape[1],featimg.shape[2],3))
    
    for row in np.arange(6,featimg.shape[1]-6,1):
        print row
        for col in np.arange(6,featimg.shape[2]-6,1):
            imgseg = featimg[:,row:row+6,col:col+6].T
#            imgseg = featimg[::-1,row:row+6,col:col+6].T
            
            feat1d = imgseg.reshape(576)
    #        feat1d = np.zeros(576)
            
#            cc = 0
    #        for ii in np.arange(0,imgseg.shape[0], 1 ):
    #        for ii in np.arange(imgseg.shape[0]-1,0, -1 ):
    #            for jj in range(imgseg.shape[1]):
    #                for kk in range(imgseg.shape[2]):
    #                    feat1d[cc] = imgseg[ii,kk,jj]
    #    #                feat1d = featimg[::-1,row:row+6,col:col+6].reshape(576)
    #                    cc += 1
            a = np.inner(feat1d,ipfilt)/np.sum(np.inner(feat1d,ipfilt))
            outimg[row,col,:] = np.exp(a)/np.sum(np.exp(a))# softmax
    
#    plt.imshow(outimg)
    cv2.imwrite(outfolder+np.str(count)+'.tif',outimg)
    print np.str(count),' classification done!'
    count += 1
#cc = 0
#for ii in np.arange(0,imgseg.shape[0], 1 ):
#for ii in np.arange(imgseg.shape[0]-1,0, -1 ):
#
#    for jj in range(imgseg.shape[1]):
#        for kk in range(imgseg.shape[2]):
#            feat1d[cc] = imgseg[ii,jj,kk]
#            cc += 1


