# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:16:02 2015

@author: saj
"""



import pickle

import scipy as sp
from scipy.signal import convolve
import time
#import sys
import os
import matplotlib.pyplot as plt
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
#import Image
import sys
sys.path.append('/Users/sajithks/Documents/deep/deeptraing/')
sys.path.append('/home/saj/Documents/deep/deeptraing/')
import fcnn
import time
import sklearn
import random
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as ab
from sklearn import svm
from multiprocessing import Pool
import itertools 
print 'libraries loaded'
import string

#%

start = time.time()

#netfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/caffe_net/ver2_1000/'
netfolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/caffe_net/ver4/'

netfiles = sorted(glob.glob(netfolder+'*.p' ))
#%%
for netcount in netfiles:
    
    caffenet = pickle.load( open( netcount, "rb" ) )
#caffenet = pickle.load( open( netfolder+'neutro_conv_222.p', "rb" ) )

#%

    print 'test data loading ...'

    imgfolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/sampimg/'
    outfolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/output/randomforest/ver3/'
    outfolder2 = '/home/saj/Documents/deep/deeptraing/data_neutrophils/output/neuralnet/ver4/'

    inputimgfiles = sorted(glob.glob(imgfolder + '*.tif'))
#    for infile in inputimgfiles:

#        orimg = cv2.imread(infile, cv2.CV_LOAD_IMAGE_UNCHANGED)
    orimg = cv2.imread(inputimgfiles[0], cv2.CV_LOAD_IMAGE_UNCHANGED)

    labelimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/ilastik/Labels00.tif', cv2.CV_LOAD_IMAGE_UNCHANGED)

#    plt.imshow(outfeatimg[:,:,1])

#%%######################### training ########################################


    featimg, maxoutl3 = fcnn.extractFcnnFeature(orimg, caffenet)


    rowdiff = orimg.shape[0]-featimg.shape[0]
    coldiff = orimg.shape[1]-featimg.shape[1]

    WINDOW = 62
    #inimg = orimg[rowdiff/2:orimg.shape[0]-rowdiff/2, coldiff/2:orimg.shape[1]-coldiff/2 ]
    labelimg = labelimg[rowdiff/2:orimg.shape[0]-rowdiff/2, coldiff/2:orimg.shape[1]-coldiff/2 ]
    
    
    bgcoord = np.argwhere(labelimg==3)
    cellcoord = np.argwhere(labelimg==1)
    cencoord = np.argwhere(labelimg==2)
    
    #%
    cellc = []
    for ii in cellcoord:
        if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
            cellc.append(ii)
            
    bagc = []
    for ii in bgcoord:
        if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
            bagc.append(ii)
    cenc = []
    for ii in cencoord:
        if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
            cenc.append(ii)
    #%       
    shuffle(cellc)
    shuffle(bagc)
    shuffle(cenc)
    cellc = cellc[0:5000]
    bagc = bagc[0:5000]
    #cenc = cenc[0:5000]


#%%
    print 'training ...'
    feat = []
    lab = []
    
    for ii in range(np.shape(cellc)[0]):
        feature = featimg[cellc[ii][0]-3:cellc[ii][0]+3, cellc[ii][1]-3:cellc[ii][1]+3,:]
        feature1d = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2])    
        feat.append(feature1d)
        lab.append(1)
#        lab.append(labelimg[cellc[ii][0], cellc[ii][1]])
    
    for ii in range(np.shape(bagc)[0]):    
        feature = featimg[bagc[ii][0]-3:bagc[ii][0]+3, bagc[ii][1]-3:bagc[ii][1]+3,:]
        feature1d = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2])    
        feat.append(feature1d)
        lab.append(2)        
#        lab.append(labelimg[bagc[ii][0], bagc[ii][1]])
    
    for ii in range(np.shape(cenc)[0]):
        feature = featimg[cenc[ii][0]-3:cenc[ii][0]+3, cenc[ii][1]-3:cenc[ii][1]+3,:]
        feature1d = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2])    
        feat.append(feature1d)
        lab.append(3)        
#        lab.append(labelimg[cenc[ii][0], cenc[ii][1]])
    
    
    
    feat = np.array(feat)
    lab = np.array(lab)
    
    rforest = rf(n_estimators=200)
    
    rforest.fit(feat,lab)


#%%######################## testing ###########################################
    count = 0

    for infile in inputimgfiles:
    
        start = time.time()
        testimg = cv2.imread(infile, cv2.CV_LOAD_IMAGE_UNCHANGED)
    #    testimg = cv2.imread(inputimgfiles[1], cv2.CV_LOAD_IMAGE_UNCHANGED)
        
#        featimg, maxoutl3 = fcnn.extractFcnnFeature(testimg, caffenet)
#        
#        classimg = np.zeros((featimg.shape[0],featimg.shape[1],3))
#        
#        for ii in np.arange(3, featimg.shape[0]-3,1):
#        #    print ii
#            groupfeat = []
#            for jj in np.arange(3, featimg.shape[1]-3,1):
#                
#                feature = featimg[ii-3:ii+3, jj-3:jj+3,:]
#                feature1d = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2])    
#                groupfeat.append(feature1d)
#                
#            classimg[ii,3:featimg.shape[1]-3,:] = rforest.predict_proba(groupfeat)
        
        savename = string.split(string.split(netcount,'/')[-1], '.')[0]
        savename = savename +'_'+ string.split(string.split(infile,'/')[-1], '.')[0]

    #    plt.imshow(classimg)
#        cv2.imwrite(outfolder+savename+'.tif', np.uint8(classimg*255))
#        count += 1
        neuralout = fcnn.classifyFcnnFeature(orimg, caffenet)
        
        cv2.imwrite(outfolder2+savename+'.tif', np.uint8(neuralout*255))

        print time.time()-start


#%%

start = time.time()
neuralout = fcnn.classifyFcnnFeature(orimg, caffenet)
print time.time()-start
plt.figure()
plt.imshow(neuralout)



count += 1



























