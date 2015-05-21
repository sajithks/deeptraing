# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:54:27 2015

@author: root
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
sys.path.append('/Users/sajithks/Documents/deeptraing')

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


#%

start = time.time()

netfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/caffe_net/ver3/'


caffenet = pickle.load( open( netfolder+'neutro_conv_1_2_4.p', "rb" ) )

#%

print 'test data loading ...'

imgfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/sampimg/'
outfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/randomforest/conv_222/'

inputimgfiles = sorted(glob.glob(imgfolder + '*.tif'))
inputimgfiles = inputimgfiles[0:2]
#for infile in inputimgfiles:

orimg = cv2.imread(inputimgfiles[0], cv2.CV_LOAD_IMAGE_UNCHANGED)

labelimg = cv2.imread('/Users/sajithks/Documents/deeptraing/data_neutrophils/ilastik/Labels00.tif', cv2.CV_LOAD_IMAGE_UNCHANGED)

#    plt.imshow(outfeatimg[:,:,1])

#%%######################### training ########################################


featimg, maxoutl3 = fcnn.extractFcnnFeature(orimg, caffenet)


rowdiff = orimg.shape[0]-featimg.shape[0]
coldiff = orimg.shape[1]-featimg.shape[1]

WINDOW = 60
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
cenc = cenc[0:5000]


#%%
print 'training ...'
feat = []
lab = []

for ii in range(np.shape(cellc)[0]):
    feature = featimg[cellc[ii][0]-3:cellc[ii][0]+3, cellc[ii][1]-3:cellc[ii][1]+3]
    feature1d = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2])    
    feat.append(feature1d)
#    lab.append(labelimg[cellc[ii][0], cellc[ii][1]])
    lab.append(1)

for ii in range(np.shape(bagc)[0]):    
    feature = featimg[bagc[ii][0]-3:bagc[ii][0]+3, bagc[ii][1]-3:bagc[ii][1]+3]
    feature1d = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2])    
    feat.append(feature1d)
#    lab.append(labelimg[bagc[ii][0], bagc[ii][1]])
    lab.append(2)

for ii in range(np.shape(cenc)[0]):
    feature = featimg[cenc[ii][0]-3:cenc[ii][0]+3, cenc[ii][1]-3:cenc[ii][1]+3]
    feature1d = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2])    
    feat.append(feature1d)
#    lab.append(labelimg[cenc[ii][0], cenc[ii][1]])
    lab.append(3)



feat = np.array(feat)
lab = np.array(lab)

rforest = rf(n_estimators=200, criterion='entropy', max_features= 0.5,max_depth=5,bootstrap=True)
#rforest = rf(n_estimators=200, criterion='entropy', max_features='log2',max_depth=5,bootstrap=True)


rforest.fit(feat,lab)


#%%######################## testing ###########################################
count = 0

for infile in inputimgfiles:

    start = time.time()
    testimg = cv2.imread(infile, cv2.CV_LOAD_IMAGE_UNCHANGED)
#    testimg = cv2.imread(inputimgfiles[1], cv2.CV_LOAD_IMAGE_UNCHANGED)
    
    featimg, maxoutl3 = fcnn.extractFcnnFeature(testimg, caffenet)
    
    classimg = np.zeros((featimg.shape[0],featimg.shape[1],3))
    
    for ii in np.arange(3, featimg.shape[0]-3,1):
    #    print ii
        groupfeat = []
        for jj in np.arange(3, featimg.shape[1]-3,1):
            
            feature = featimg[ii-3:ii+3, jj-3:jj+3]
            feature1d = feature.reshape(feature.shape[0]*feature.shape[1]*feature.shape[2])    
            groupfeat.append(feature1d)
            
        classimg[ii,3:featimg.shape[1]-3,:] = rforest.predict_proba(groupfeat)
    
    
    plt.figure(),plt.imshow(classimg)
#    cv2.imwrite(outfolder+np.str(count+10000000000)+'.png', np.uint8(classimg*255))
    count += 1

    print time.time()-start


#%%
#
#start = time.time()
#neuralout = fcnn.classifyFcnnFeature(orimg, caffenet)
#print time.time()-start
#plt.figure()
#plt.imshow(neuralout)




#count += 1



























