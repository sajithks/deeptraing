# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:35:28 2015

@author: Sajith
"""



import scipy as sp
import time
#import sys
#import os
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
#import fiteli
#from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool as Pool
#from numba import jit, double
#import cmath
#from skimage import transform
#import scipy.ndimage
#import pymeanshift as pms
#import morphsnakes
#import criticalpoints as cr
#import skimage
#from skimage.morphology import watershed
#import random
#from random import randrange

WIN_SIZE = 15
WINDOW = 2*WIN_SIZE + 1

#%% training set
orimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/img/Aligned0000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
orimg = np.float32(orimg)
orimg = orimg-orimg.min()
orimg = np.uint8(255*(orimg/orimg.max()))

segimage = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/seg/labeledAligned0000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
cenimage = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/seg/cenAligned0000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#segimage = cv2.imread('/home/saj/Documents/deeptraing-master/data/seg/labeled20141021_ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#fgcoord = np.argwhere(segimage>0)
fgimage = segimage>0
cellimage = (fgimage* ~(cenimage==1))

bgcoord = np.argwhere(segimage==0)
cellcoord = np.argwhere(cellimage==1)
cencoord = np.argwhere(cenimage==1)

#%%
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
        
shuffle(cellc)
shuffle(bagc)
shuffle(cenc)
cellc = cellc[0:20000]
bagc = bagc[0:20000]
#cenc = cenc[0:200]

#%%
#st = time.time()
#countval = 0
        
# training data

outfolder = '/home/saj/Downloads/caffe-master/examples/ecoli/neutrotraindata/'
strlist = []

#cellregion
for ii in cellc:
    savname = 'cell_' + np.str(ii[0]) +'_' + np.str(ii[1]) + '.png'
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    cv2.imwrite(outfolder + savname, savimg)
    strval = outfolder + savname + " "+ "0"
    strlist.append(strval)
#bagground region
bagccname = []
for ii in bagc:
    savname = 'bg_' + np.str(ii[0]) +'_' + np.str(ii[1]) + '.png'
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    cv2.imwrite(outfolder + savname, savimg)
    strval = outfolder + savname + " "+ "1"
    strlist.append(strval)
#cell center region
#labcen, ncc = label(cenimage==1, np.ones((3,3)))

for ii in cenc:
    cencrop = orimg[ii[0] - WINDOW:ii[0] + WINDOW, ii[1] - WINDOW:ii[1] + WINDOW]
    for ang in np.arange(0,350,200):
        rotimg = np.uint8(interpolation.rotate(cencrop,ang,reshape=False))
        savimg = rotimg[rotimg.shape[0]/2 - WIN_SIZE:rotimg.shape[0]/2 + WIN_SIZE, rotimg.shape[1]/2 - WIN_SIZE:rotimg.shape[1]/2 + WIN_SIZE]
        savname = 'cen_' + np.str(ii[0]) +'_' + np.str(ii[1]) + np.str(ang) +'.png'
        cv2.imwrite(outfolder + savname, savimg)
        strval = outfolder + savname + " "+ "2"
        strlist.append(strval)
#print time.time() -st
shuffle(strlist)
target = open(outfolder +'training', 'w')        

trainsize = np.shape(strlist)[0]-np.mod(np.shape(strlist)[0],100)
for ii in range(trainsize):
    target.write(strlist[ii])
    target.write("\n")
target.close()

#%%






#%% #################### testing set ########################################
orimg = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/img/Aligned0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = cv2.imread('/home/saj/Documents/deeptraing-master/data/img/ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
orimg = np.float32(orimg)
orimg = orimg-orimg.min()
orimg = np.uint8(255*(orimg/orimg.max()))

segimage = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/seg/labeledAligned0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
cenimage = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/seg/cenAligned0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#segimage = cv2.imread('/home/saj/Documents/deeptraing-master/data/seg/labeled20141021_ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#fgcoord = np.argwhere(segimage>0)
fgimage = segimage>0
cellimage = (fgimage* ~(cenimage==1))

bgcoord = np.argwhere(segimage==0)
cellcoord = np.argwhere(cellimage==1)
cencoord = np.argwhere(cenimage==1)

#%%
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

shuffle(cellc)
shuffle(bagc)
shuffle(cenc)
cellc = cellc[0:2000]
bagc = bagc[0:2000]
cenc = cenc[0:2000]

#%
#st = time.time()
#countval = 0
        
# training data

outfolder = '/home/saj/Downloads/caffe-master/examples/ecoli/neutrotestdata/'
strlist = []

#cellregion
for ii in cellc:
    savname = 'cell_' + np.str(ii[0]) +'_' + np.str(ii[1]) + '.png'
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    cv2.imwrite(outfolder + savname, savimg)
    strval = outfolder + savname + " "+ "0"
    strlist.append(strval)
#bagground region
bagccname = []
for ii in bagc:
    savname = 'bg_' + np.str(ii[0]) +'_' + np.str(ii[1]) + '.png'
    savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
    cv2.imwrite(outfolder + savname, savimg)
    strval = outfolder + savname + " "+ "1"
    strlist.append(strval)
#cell center region
#labcen, ncc = label(cenimage==1, np.ones((3,3)))

for ii in cenc:
        savimg = orimg[ii[0] - WIN_SIZE:ii[0] + WIN_SIZE, ii[1] - WIN_SIZE:ii[1] + WIN_SIZE]
        savname = 'cen_' + np.str(ii[0]) +'_' + np.str(ii[1]) + np.str(ang) +'.png'
        cv2.imwrite(outfolder + savname, savimg)
        strval = outfolder + savname + " "+ "2"
        strlist.append(strval)
#print time.time() -st
shuffle(strlist)

testsize = np.shape(strlist)[0]-np.mod(np.shape(strlist)[0],100)
target = open(outfolder +'testing', 'w')        

for ii in range(testsize):
    target.write(strlist[ii])
    target.write("\n")
target.close()




