# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:26:53 2015

@author: saj
"""

import scipy as sp
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import Image

# Make sure that caffe is on the python path:
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
#%%
# Load the net, list its data and params, and filter an example image.
caffe.set_mode_cpu()
#net = caffe.Net(caffe_root + 'examples/ecoli/ecolifile2deploy.prototxt',
#                caffe_root + 'examples/ecoli/file8bit2_iter_10000.caffemodel',
#                caffe.TEST)
net = caffe.Net(caffe_root + 'examples/ecoli/ecolifile2deploy.prototxt',
                caffe_root + 'examples/ecoli/neutro3clas_iter_10000.caffemodel',
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
st = time.time()
pval = ['conv1', 'conv2', 'conv3']
c = 0
featmat = []
filtermat = []
for kk in pval:
    for ii in range(np.shape(net.params[kk][0].data)[0]):
        for jj in range(np.shape(net.params[kk][0].data[ii])[0]):
            filt = net.params[kk][0].data[ii][jj]
#            filtermat.append(net.params['conv1'][0].data[ii][jj])
            featmat.append(sp.ndimage.convolve(orimg,filt))
            
print time.time()-st
featmat = np.array(featmat)
print 'features extracted'
#%%
print 'training ...'
feat = []
lab = []

#for ii in range(10000):
#    row = random.randint(0,orimg.shape[0]-1)
#    col = random.randint(0,orimg.shape[1]-1)
#    feat.append(featmat[:,row,col])
#    lab.append(labimg[row,col])
for row in np.int32(np.linspace(0, orimg.shape[0]-1, 100)):
    for col in np.int32(np.linspace(0, orimg.shape[1]-1, 100)):
        feat.append(featmat[:, row, col])
        lab.append(labimg[row, col])
    
feat = np.array(feat)
lab = np.array(lab)

rforest = rf(n_estimators=200)

rforest.fit(feat,lab)
print 'training done.'
#%%
#st = time.time()
#adaboost = ab(n_estimators=200)
#adaboost.fit(feat,lab)
#print time.time()-st
#%%
#st = time.time()
#svmclas = svm.SVC()
#svmclas.fit(feat,lab)
#print time.time()-st
#%%
############################### testing ###############################
print 'test data loading ...'
st = time.time()
inimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/img/Aligned0005.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#inimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0004.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#inimg = np.float32(inimg)
#inimg = inimg-inimg.min()
#inimg = np.uint8(255*(inimg/inimg.max()))

pval = ['conv1', 'conv2', 'conv3']
c = 0
print 'test data loaded'

print 'feature extracting...'

featmat = []
for kk in pval:
    for ii in range(np.shape(net.params[kk][0].data)[0]):
        for jj in range(np.shape(net.params[kk][0].data[ii])[0]):
            filt = net.params[kk][0].data[ii][jj]     
            featmat.append(sp.ndimage.convolve(inimg,filt))
            
print time.time()-st
featmat = np.array(featmat)
print 'feature extraction done'
#%%
print 'start classification...'
st = time.time()
outimg = np.zeros((inimg.shape[0], inimg.shape[1]))
#featreshape = featmat.reshape(featmat.shape[0],featmat.shape[1]*featmat.shape[2]).T
#outimg = rforest.predict(featreshape).reshape(featmat.shape[1],featmat.shape[2])

for ii in range(inimg.shape[1]):
    print ii
#    outimg[ii,:] = rforest.predict(featmat[:,ii,:].T)    
#    outimg[ii,:] = adaboost.predict_proba(featmat[:,ii,:].T)[:,0]
    outimg[:,ii] =rforest.predict_proba(featmat[:,:,ii].T)[:,2]
#    outimg[ii,:] = svmclas.predict_proba(featmat[:,ii,:].T)[:,0]

  
print 'testing time ', time.time() - st

plt.imshow(outimg)
print 'classification done!'



