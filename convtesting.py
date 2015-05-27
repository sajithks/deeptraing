# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:23:52 2015

@author: root
"""
import pickle

import scipy as sp
from scipy.signal import convolve
import time
#import sys
import os
import matplotlib.pyplot as plt
sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf')
sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/ecoli_det_seg')
import createbacteria  as cb
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
import string


def fftConvolve(img1, img2):
    
    maxrow = np.max([img1.shape[0], img2.shape[0]])
    maxcol = np.max([img1.shape[1], img2.shape[1]])
    
    templateimg1 = np.zeros((maxrow, maxcol))
    templateimg2 = np.zeros((maxrow, maxcol))
    
    difrow1 = maxrow - img1.shape[0]
    difcol1 = maxcol - img1.shape[1]
    
    templateimg1[difrow1/2:difrow1/2+img1.shape[0], difcol1/2:difcol1/2 + img1.shape[1]]  = img1  
    
    difrow2 = maxrow - img2.shape[0]
    difcol2 = maxcol - img2.shape[1]    
    templateimg2[difrow2/2:difrow2/2+img2.shape[0], difcol2/2:difcol2/2 + img2.shape[1]]  = img2  
    
        
    fftimg1 = np.fft.fft2(templateimg1)
    fftimg2 = np.fft.fft2(templateimg2)
    
    outfft = fftimg1*fftimg2
    
    return(np.real( fft.ifft2(outfft)))
    

#%%

netfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/caffe_net/'
caffenet = pickle.load( open( netfolder+'neutro_conv_111.p', "rb" ) )

f1 = caffenet['conv3_filt']
b1 = caffenet['conv3_bias']

f2 = caffenet['conv3_level2_filt']
b2 = caffenet['conv3_level2_bias']

f3 = caffenet['conv3_level3_filt']
b3 = caffenet['conv3_level3_bias']
    
#convolve(flipFilter() , flipFilter(f2[0][0]) ,'full')
cb.myshow(f1[0][0])
cb.myshow(f2[0][0])
cb.myshow(f3[0][0])

f1conf2 = convolve(np.fliplr(np.flipud(f1[0][0])), np.fliplr(np.flipud(f2[0][0])),'full')
cb.myshow(f1conf2)

f1f2f3 = convolve(np.fliplr(np.flipud(f1conf2)), np.fliplr(np.flipud(f3[0][0])), 'full'  )
cb.myshow(f1f2f3)
a = cv2.imread('/Users/sajithks/Documents/deeptraing/data_neutrophils/img/Aligned0000.tif',-1 )

convout =  convolve( a,np.fliplr(np.flipud(f1f2f3)), 'valid')
#cb.myshow2(convout)
b1f2f3 =  convolve(np.fliplr(np.flipud(f2[0][0])), np.fliplr(np.flipud(f3[0][0])),'full')*b1

#%%
f1f2 = fftConvolve(np.fliplr(np.flipud(f1[0][0])), np.fliplr(np.flipud(f2[0][0])))
f1f2f3 = fftConvolve(np.fliplr(np.flipud(f1f2)), np.fliplr(np.flipud(f3[0][0])))

af1f2f3 = fftConvolve(a, np.fliplr(np.flipud( f1f2f3)))

b1f2f3 = fftConvolve(np.fliplr(np.flipud(f2[0][0])), np.fliplr(np.flipud(f3[0][0])))*b1

b2f3 = b2*f3





cb.myshow2(np.real(af1f2f3))



