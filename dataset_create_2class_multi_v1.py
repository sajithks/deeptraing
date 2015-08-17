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
WIN_SIZE = 31
WINDOW = 2*WIN_SIZE + 1


#outfolder = '/home/saj/Downloads/caffe-master/examples/ecoli/neutrotraindata/'
#traindataloc = '/home/saj/Documents/cbalinux/deeplearning/caffe-master/examples/ecoli/neutrotraindata/'
#testdataloc = '/home/saj/Documents/cbalinux/deeplearning/caffe-master/examples/ecoli/neutrotestdata/'
traindataloc = '/home/saj/Downloads/caffe-master/examples/neutrophiles/neutrotraindata/'
testdataloc = '/home/saj/Downloads/caffe-master/examples/neutrophiles/neutrotestdata/'

#% training set
# read file names
print 'reading file names ...'
#datafolder = '/home/saj/Documents/cbalinux/deeplearning/programs/data_neutrophils/labels/'
datafolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/'
rawimgfiles = sorted(glob.glob(datafolder +'sampimg/' + '*.tif'))
labelfiles = sorted(glob.glob(datafolder +'labels/fulllabels/' + '*.png'))[0:2]
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
orimg = np.uint8(255*(orimg/orimg.max()))

labelimg = cv2.imread(labelfiles[0], cv2.CV_LOAD_IMAGE_UNCHANGED)

print 'read training image and label image '

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
#cenc = []
#for ii in cencoord:
#    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
#        cenc.append(ii)
#%       
shuffle(cellc)
shuffle(bagc)
#shuffle(cenc)
cellc = cellc[0:5000]
bagc = bagc[0:5000]
#cenc = cenc[0:715]

#%%
#st = time.time()
#countval = 0
        
# training data
print 'writing training images ...'
strlist = []
cval = 0
#cellregion
for ii in cellc:
    savname = 'cell_' +np.str(cval)+'_' + np.str(ii[0]) +'_' + np.str(ii[1]) + '.png'
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    cv2.imwrite(traindataloc + savname, savimg)
    strval = savname + " "+ "0"
#    strval = traindataloc + savname + " "+ "0"
   
    strlist.append(strval)
#bagground region
bagccname = []
for ii in bagc:
    savname = 'bg_' + np.str(cval) + '_' + np.str(ii[0]) +'_' + np.str(ii[1]) + '.png'
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    cv2.imwrite(traindataloc + savname, savimg)
    strval = savname + " "+ "1"
#    strval = traindataloc + savname + " "+ "1"
    
    strlist.append(strval)
#cell center region
#labcen, ncc = label(cenimage==1, np.ones((3,3)))

#for ii in cenc:
#    cencrop = orimg[ii[0] - WINDOW:ii[0] + WINDOW, ii[1] - WINDOW:ii[1] + WINDOW]
#    for ang in np.arange(0,350,400):
#        rotimg = np.uint8(interpolation.rotate(cencrop,ang,reshape=False))
#        savimg = rotimg[rotimg.shape[0]/2 - WIN_SIZE:rotimg.shape[0]/2 + WIN_SIZE, rotimg.shape[1]/2 - WIN_SIZE:rotimg.shape[1]/2 + WIN_SIZE]
#        savname = 'cen_' + np.str(cval) + '_' + np.str(ii[0]) +'_' + np.str(ii[1]) + np.str(ang) +'.png'
#        cv2.imwrite(traindataloc + savname, savimg)
#        strval = traindataloc + savname + " "+ "2"
#        strlist.append(strval)
##print time.time() -st
shuffle(strlist)
target = open(traindataloc +'training', 'w')        

trainsize = np.shape(strlist)[0]-np.mod(np.shape(strlist)[0],100)
for ii in range(trainsize):
    target.write(strlist[ii])
    target.write("\n")
target.close()
print 'training images wrote'
#%






#% #################### testing set ########################################
print 'reading testing images ...'
orimg = cv2.imread(rawimgfiles[1], cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
orimg = np.float32(orimg)
orimg = orimg-orimg.min()
orimg = np.uint8(255*(orimg/orimg.max()))

labelimg = cv2.imread(labelfiles[1], cv2.CV_LOAD_IMAGE_UNCHANGED)


bgcoord = np.argwhere(labelimg==3)
cellcoord = np.argwhere(labelimg==1)
#cencoord = np.argwhere(labelimg==2)
print 'testing images read'
#%
cellc = []
for ii in cellcoord:
    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
        cellc.append(ii)
        
bagc = []
for ii in bgcoord:
    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
        bagc.append(ii)
#cenc = []
#for ii in cencoord:
#    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
#        cenc.append(ii)

shuffle(cellc)
shuffle(bagc)
#shuffle(cenc)
cellc = cellc[0:5000]
bagc = bagc[0:5000]
#cenc = cenc[0:2000]

#%
#st = time.time()
#countval = 0
        
# training data
print 'writing testing images ...'
#outfolder = '/home/saj/Downloads/caffe-master/examples/ecoli/neutrotestdata/'
strlist = []

cval = 1
#cellregion
for ii in cellc:
    savname = 'cell_' + np.str(cval)+'_' + np.str(ii[0]) +'_' + np.str(ii[1]) + '.png'
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    cv2.imwrite(testdataloc + savname, savimg)
    strval = savname + " "+ "0"
#    strval = testdataloc + savname + " "+ "0"
    
    strlist.append(strval)
#bagground region
bagccname = []
for ii in bagc:
    savname = 'bg_' + np.str(cval)+'_' + np.str(ii[0]) +'_' + np.str(ii[1]) + '.png'
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    cv2.imwrite(testdataloc + savname, savimg)
    strval = savname + " "+ "1"
#    strval = testdataloc + savname + " "+ "1"
    
    strlist.append(strval)
#cell center region
#labcen, ncc = label(cenimage==1, np.ones((3,3)))

#for ii in cenc:
#    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
#    savname = 'cen_' + np.str(cval) + '_' + np.str(ii[0]) +'_' + np.str(ii[1]) + np.str(ang) +'.png'
#    cv2.imwrite(testdataloc + savname, savimg)
#    strval = testdataloc + savname + " "+ "2"
#    strlist.append(strval)
#print time.time() -st
shuffle(strlist)

testsize = np.shape(strlist)[0]-np.mod(np.shape(strlist)[0],100)
target = open(testdataloc +'testing', 'w')        

for ii in range(testsize):
    target.write(strlist[ii])
    target.write("\n")
target.close()
print 'wrote testing images'
###############################################################################
############################# data set creation done ##########################
#%% training pre built deep network

#cmd = "sh examples/ecoli/train_ecoli.sh"
#p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#while True:
#    out = p.stderr.read(1)
#    if out == '' and p.poll() != None:
#        break
#    if out != '':
#        sys.stdout.write(out)
#        sys.stdout.flush()
#p.wait()
##%% 
##print 'network training done'
################################################################################
#
#
#caffe.set_mode_cpu()
##net = caffe.Net(caffe_root + 'examples/ecoli/ecolifile2deploy.prototxt',
##                caffe_root + 'examples/ecoli/file8bit2_iter_10000.caffemodel',
##                caffe.TEST)
#net = caffe.Net(caffe_root + 'examples/ecoli/neutro3classv3_deploy.prototxt',
#                caffe_root + 'examples/ecoli/neutro3clasv3_iter_10000.caffemodel',
#                caffe.TEST)
#
######################### training #######################################
##%% read images
#print 'loading image and label ...'
#orimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/img/Aligned0000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
##orimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0003.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = np.float32(orimg)
#orimg = orimg-orimg.min()
#orimg = np.uint8(255*(orimg/orimg.max()))
#
##segimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/seg/labeled20141021_ex_Phase0003.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#segimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/seg/labeledAligned0000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#
#
#labimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/seg/Aligned00003class.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#
#print('image and label loaded')
##%%
#print 'extracting features ...'
#st = time.time()
#pval = ['conv1', 'conv2', 'conv3']
#c = 0
#featmat = []
#filtermat = []
#for kk in pval:
#    for ii in range(np.shape(net.params[kk][0].data)[0]):
#        for jj in range(np.shape(net.params[kk][0].data[ii])[0]):
#            filt = net.params[kk][0].data[ii][jj]
##            filtermat.append(net.params['conv1'][0].data[ii][jj])
#            featmat.append(sp.ndimage.convolve(orimg,filt))
#            
#print time.time()-st
#featmat = np.array(featmat)
#print 'features extracted'
##%%
#print 'training ...'
#feat = []
#lab = []
#
##for ii in range(10000):
##    row = random.randint(0,orimg.shape[0]-1)
##    col = random.randint(0,orimg.shape[1]-1)
##    feat.append(featmat[:,row,col])
##    lab.append(labimg[row,col])
#for row in np.int32(np.linspace(0, orimg.shape[0]-1, 100)):
#    for col in np.int32(np.linspace(0, orimg.shape[1]-1, 100)):
#        feat.append(featmat[:, row, col])
#        lab.append(labimg[row, col])
#    
#feat = np.array(feat)
#lab = np.array(lab)
#
#rforest = rf(n_estimators=200)
#
#rforest.fit(feat,lab)
#print 'training done.'
##%%
#
##%%
################################ testing ###############################
#print 'test data loading ...'
#st = time.time()
#inimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/img/Aligned0005.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
##inimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0004.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
##inimg = np.float32(inimg)
##inimg = inimg-inimg.min()
##inimg = np.uint8(255*(inimg/inimg.max()))
#
#pval = ['conv1', 'conv2', 'conv3']
#c = 0
#print 'test data loaded'
#
#print 'feature extracting...'
#
#featmat = []
#for kk in pval:
#    for ii in range(np.shape(net.params[kk][0].data)[0]):
#        for jj in range(np.shape(net.params[kk][0].data[ii])[0]):
#            filt = net.params[kk][0].data[ii][jj]     
#            featmat.append(sp.ndimage.convolve(inimg,filt))
#            
#print time.time()-st
#featmat = np.array(featmat)
#print 'feature extraction done'
##%%
#print 'start classification...'
#st = time.time()
#outimg = np.zeros((inimg.shape[0], inimg.shape[1]))
##featreshape = featmat.reshape(featmat.shape[0],featmat.shape[1]*featmat.shape[2]).T
##outimg = rforest.predict(featreshape).reshape(featmat.shape[1],featmat.shape[2])
#
#for ii in range(inimg.shape[1]):
#    print ii
##    outimg[ii,:] = rforest.predict(featmat[:,ii,:].T)    
##    outimg[ii,:] = adaboost.predict_proba(featmat[:,ii,:].T)[:,0]
#    outimg[:,ii] =rforest.predict_proba(featmat[:,:,ii].T)[:,2]
##    outimg[ii,:] = svmclas.predict_proba(featmat[:,ii,:].T)[:,0]
#
#  
#print 'testing time ', time.time() - st
#
#plt.imshow(outimg)
#print 'classification done!'















