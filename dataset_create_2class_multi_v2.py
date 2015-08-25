# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:09:31 2015

@author: saj
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
#caffe_root = '/home/saj/Downloads/caffelatest/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

#import caffe
import time
import sklearn
import random
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as ab
from sklearn import svm
from multiprocessing import Pool
import h5py


print 'libraries loaded'
WIN_SIZE = 30
WINDOW = 2*WIN_SIZE + 1
WIN_SIZE2 = 9

#outfolder = '/home/saj/Downloads/caffe-master/examples/ecoli/neutrotraindata/'
#traindataloc = '/home/saj/Documents/cbalinux/deeplearning/caffe-master/examples/ecoli/neutrotraindata/'
#testdataloc = '/home/saj/Documents/cbalinux/deeplearning/caffe-master/examples/ecoli/neutrotestdata/'
traindataloc = 'examples/neutrophiles/neutrotraindatah5/'
testdataloc = 'examples/neutrophiles/neutrotestdatah5/'

#% training set
# read file names
print 'reading file names ...'
#datafolder = '/home/saj/Documents/cbalinux/deeplearning/programs/data_neutrophils/labels/'
datafolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/'
rawimgfiles = sorted(glob.glob(datafolder +'sampimg/' + '*.tif'))[0:2]
labelfiles = sorted(glob.glob(datafolder +'labels/cell/' + '*.png'))[0:2]
print 'file name read'

#%% clear temporary locations of training and test data

print ' clearing training and testing data ... '
if(os.system('rm -rf '+traindataloc) ==0):print 'traindataloc removed'
if(os.system('mkdir '+traindataloc) ==0):print 'traindataloc recreated'

if(os.system('rm -rf '+testdataloc) ==0):print 'testdataloc removed'
if(os.system('mkdir '+testdataloc) ==0):print 'testdataloc recreated'
#os.system('rm -rf '+traindataloc+'bg*')
#os.system('rm -rf '+traindataloc+'cen*')

#os.system('rm -rf '+testdataloc+'cell*')
#os.system('rm -rf '+testdataloc+'bg*')
#os.system('rm -rf '+testdataloc+'cen*')
print 'cleared training and testing data '
#%##############################################################################
print 'reading training image and label image ...'
orimg = cv2.imread(rawimgfiles[0], cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
orimg = np.float32(orimg)
orimg = orimg-orimg.min()
orimg = orimg/orimg.max()

labelimg = cv2.imread(labelfiles[0], cv2.CV_LOAD_IMAGE_UNCHANGED)
labelimg = np.float32(labelimg)

print 'read training image and label image '

bgcoord = np.argwhere(labelimg==0)
cellcoord = np.argwhere(labelimg==1)
#cencoord = np.argwhere(labelimg==2)

#%
cellc = []
for ii in cellcoord:
    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
        cellc.append(ii)
        
bagc = []
for ii in bgcoord:
    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
        bagc.append(ii)
      
shuffle(cellc)
shuffle(bagc)
#shuffle(cenc)
cellc = cellc[0:5000]
bagc = bagc[0:5000]
cellc = cellc + bagc
#cenc = cenc[0:715]

#%% training data
print 'writing training images ...'
strlist = []
cval = 0
#cellregion

filelist = []
indexval = 0
for ii in cellc:
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    savlab = labelimg[ii[0] - WIN_SIZE2:ii[0] + WIN_SIZE2, ii[1] - WIN_SIZE2:ii[1] + WIN_SIZE2]
    filename = caffe_root + traindataloc + 'trainhdf%d.h5' % indexval
    relfilename = traindataloc + 'trainhdf%d.h5' % indexval
#    filename = traindataloc +'trainhdf%d.h5' % indexval
    
    with h5py.File(filename, 'w') as f:
        f['data1'] = savimg[np.newaxis,np.newaxis,:,:]
        f['data2'] = savlab[np.newaxis,np.newaxis,:,:]  
    filelist.append(filename) 
    indexval += 1
#with open('trainlist.txt', 'w') as f:
with open(caffe_root + traindataloc+'trainlist.txt', 'w') as f:

    for filename in filelist:
        f.write(filename + '\n')

print 'training images wrote'
#%





#%% #################### testing set ########################################
print 'reading testing images ...'
orimg = cv2.imread(rawimgfiles[1], cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
orimg = np.float32(orimg)
orimg = orimg-orimg.min()
orimg = orimg/orimg.max()

labelimg = cv2.imread(labelfiles[1], cv2.CV_LOAD_IMAGE_UNCHANGED)
labelimg = np.float32(labelimg)

bgcoord = np.argwhere(labelimg==0)
cellcoord = np.argwhere(labelimg==1)
#cencoord = np.argwhere(labelimg==2)

#%
cellc = []
for ii in cellcoord:
    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
        cellc.append(ii)
        
bagc = []
for ii in bgcoord:
    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
        bagc.append(ii)
      
shuffle(cellc)
shuffle(bagc)
#shuffle(cenc)
cellc = cellc[0:5000]
bagc = bagc[0:5000]
cellc = cellc + bagc
#%
#st = time.time()
#countval = 0
        
# training data
print 'writing testing images ...'
#outfolder = '/home/saj/Downloads/caffe-master/examples/ecoli/neutrotestdata/'
strlist = []
cval = 0
#cellregion

filelist = []
indexval = 0
for ii in cellc:
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    savlab = labelimg[ii[0] - WIN_SIZE2:ii[0] + WIN_SIZE2, ii[1] - WIN_SIZE2:ii[1] + WIN_SIZE2]
#    filename = 'testhdf%d.h5' % indexval
    filename = caffe_root+testdataloc +'testhdf%d.h5' % indexval
    relfilename = testdataloc + 'trainhdf%d.h5' % indexval

    with h5py.File(filename, 'w') as f:
        f['data1'] = savimg[np.newaxis,np.newaxis,:,:]
        f['data2'] = savlab[np.newaxis,np.newaxis,:,:] 
    filelist.append(filename) 
    indexval += 1
#with open('testlist.txt', 'w') as f:
with open(caffe_root+testdataloc+'testlist.txt', 'w') as f:
    for filename in filelist:
        f.write(filename + '\n')

print 'wrote testing images'
###############################################################################
