# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:35:28 2015

@author: Sajith
"""



import scipy as sp
import time
import sys
import os
sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf')
sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/ecoli_det_seg')
import createbacteria  as cb
import numpy as np
from matplotlib import pyplot as plt
import skimage.morphology as skmorph
import cv2
from scipy.ndimage import label
import fiteli
from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool as Pool
#from numba import jit, double
#import cmath
from skimage import transform
import scipy.ndimage
import pymeanshift as pms
import morphsnakes
import criticalpoints as cr
import skimage
from skimage.morphology import watershed
#import random
from random import randrange

#%%
orimg = cv2.imread('/Users/sajithks/Documents/project_cell_tracking/processed/result_comparisons/tracking/phaseimage/img/ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
segimage = cv2.imread('/Users/sajithks/Documents/project_cell_tracking/processed/result_comparisons/tracking/phaseimage/Analysis/Segmentation/img/labeled20141021_ex_Phase0001.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
fgcoord = np.argwhere(segimage>0)
bgcoord = np.argwhere(segimage==0)

WIN_SIZE = 10
WINDOW = 2*WIN_SIZE + 1

#%%
fgc = []
for ii in fgcoord:
    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
        fgc.append(ii)
        
bgc = []
for ii in bgcoord:
    if(ii[0]>WINDOW and ii[1]>WINDOW and ii[0]<orimg.shape[0]-WINDOW and ii[1]<orimg.shape[1]-WINDOW):
        bgc.append(ii)
#countval = 0
        
# training data

outfolder = '/Users/sajithks/Documents/caffe_traindata/ecoli/traindata/'
target = open(outfolder +'training', 'w')        
for ii in np.unique(np.int32(np.linspace(0, min(np.shape(fgc)[0], np.shape(bgc)[0])-1, 50000 ))):
    savname = 'fg_' + np.str(fgc[ii][0]) +'_' + np.str(fgc[ii][1]) + '.png'
    savimg = orimg[fgc[ii][0] - WIN_SIZE:fgc[ii][0] + WIN_SIZE, fgc[ii][1] - WIN_SIZE:fgc[ii][1] + WIN_SIZE]
    cv2.imwrite( outfolder + savname, savimg)
    target.write(savname) 
    target.write(" ")
    target.write("1")
    target.write("\n")
#    countval = countval + 1

    savname = 'bg_' + np.str(bgc[ii][0]) +'_' + np.str(bgc[ii][1]) + '.png'
    savimg = orimg[(bgc[ii][0] - WIN_SIZE):(bgc[ii][0] + WIN_SIZE), (bgc[ii][1] - WIN_SIZE):(bgc[ii][1] + WIN_SIZE)]
    cv2.imwrite( outfolder + savname, savimg)
    target.write(savname) 
    target.write(" ") 
    target.write("0")
    target.write("\n")
#    countval = countval + 1

target.close()

#%%
        
# testing data
outfolder = '/Users/sajithks/Documents/caffe_traindata/ecoli/testdata/'
target = open(outfolder +'testing', 'w')        

for ii in np.unique(np.int32(np.linspace(0, min(np.shape(fgc)[0], np.shape(bgc)[0])-1, 1024 ))):
    savname = 'fg_' + np.str(fgc[ii][0]) +'_' + np.str(fgc[ii][1]) + '.png'
    savimg = orimg[fgc[ii][0] - WIN_SIZE:fgc[ii][0] + WIN_SIZE, fgc[ii][1] - WIN_SIZE:fgc[ii][1] + WIN_SIZE]
    cv2.imwrite( outfolder + savname, savimg)
    target.write(savname) 
    target.write(" ")
    target.write("1")
    target.write("\n")
#    countval = countval + 1

    savname = 'bg_' + np.str(bgc[ii][0]) +'_' + np.str(bgc[ii][1]) + '.png'
    savimg = orimg[(bgc[ii][0] - WIN_SIZE):(bgc[ii][0] + WIN_SIZE), (bgc[ii][1] - WIN_SIZE):(bgc[ii][1] + WIN_SIZE)]
    cv2.imwrite( outfolder + savname, savimg)
    target.write(savname) 
    target.write(" ") 
    target.write("0")
    target.write("\n")
#    countval = countval + 1

target.close()


