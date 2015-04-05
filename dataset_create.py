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
orimg = cv2.imread('/Users/sajithks/Documents/project_cell_tracking/processed/result_comparisons/tracking/growthrate_track/phase1/img/transformed_rescaled_cropped_img__000000000_150_000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
segimage = cv2.imread('/Users/sajithks/Documents/project_cell_tracking/processed/result_comparisons/tracking/growthrate_track/phase1/Analysis/Segmentation/img/transformed_rescaled_cropped_img__000000000_150_000.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
fgcoord = np.argwhere(segimage>0)
bgcoord = np.argwhere(segimage==0)

WIN_SIZE = 5
WINDOW = 2*WIN_SIZE + 1

#%% training set
outfolder = '/Users/sajithks/Documents/caffe_traindata/ecoli/'
target = open(outfolder +'training', 'w')        

for ii in range(10000):
    
    fgc = fgcoord[randrange(0, fgcoord.shape[0]), :]
    if(fgc[0]>WINDOW and fgc[1]>WINDOW and fgc[0]<orimg.shape[0]-WINDOW and fgc[1]<orimg.shape[1]-WINDOW):
        savname = 'fg_' + np.str(fgc[0]) +'_' + np.str(fgc[1]) + '.png'
        savimg = orimg[fgc[0] - WIN_SIZE:fgc[0] + WIN_SIZE, fgc[1] - WIN_SIZE:fgc[1] + WIN_SIZE]
        cv2.imwrite( outfolder + savname, savimg)
        target.write(savname) 
        target.write(" ")
        target.write("1")
        target.write("\n")
        
        
    bgc = bgcoord[randrange(0, bgcoord.shape[0]), :]
    if(bgc[0]>WINDOW and bgc[1]>WINDOW and bgc[0]<orimg.shape[0]-WINDOW and bgc[1]<orimg.shape[1]-WINDOW):
        savname = 'bg_' + np.str(fgc[0]) +'_' + np.str(fgc[1]) + '.png'
        savimg = orimg[fgc[0] - WIN_SIZE:fgc[0] + WIN_SIZE, fgc[1] - WIN_SIZE:fgc[1] + WIN_SIZE]
        cv2.imwrite( outfolder + savname, savimg)
        target.write(savname) 
        target.write(" ") 
        target.write("0")
        target.write("\n")

target.close()

#%%testing set

outfolder = '/Users/sajithks/Documents/caffe_traindata/ecoli/'
target = open(outfolder +'testing', 'w')        

for ii in range(5000):
    
    fgc = fgcoord[randrange(0, fgcoord.shape[0]), :]
    if(fgc[0]>WINDOW and fgc[1]>WINDOW and fgc[0]<orimg.shape[0]-WINDOW and fgc[1]<orimg.shape[1]-WINDOW):
        savname = 'fg_' + np.str(fgc[0]) +'_' + np.str(fgc[1]) + '.png'
        savimg = orimg[fgc[0] - WIN_SIZE:fgc[0] + WIN_SIZE, fgc[1] - WIN_SIZE:fgc[1] + WIN_SIZE]
        cv2.imwrite( outfolder + savname, savimg)
        target.write(savname) 
        target.write(" ")
        target.write("1")
        target.write("\n")
        
        
    bgc = bgcoord[randrange(0, bgcoord.shape[0]), :]
    if(bgc[0]>WINDOW and bgc[1]>WINDOW and bgc[0]<orimg.shape[0]-WINDOW and bgc[1]<orimg.shape[1]-WINDOW):
        savname = 'bg_' + np.str(fgc[0]) +'_' + np.str(fgc[1]) + '.png'
        savimg = orimg[fgc[0] - WIN_SIZE:fgc[0] + WIN_SIZE, fgc[1] - WIN_SIZE:fgc[1] + WIN_SIZE]
        cv2.imwrite( outfolder + savname, savimg)
        target.write(savname) 
        target.write(" ") 
        target.write("0")
        target.write("\n")

target.close()







