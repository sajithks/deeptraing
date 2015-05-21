# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:52:26 2015

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
import string

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

#% 
#print 'network training done'
###############################################################################
modelfolder = '/home/saj/Downloads/caffe-master/examples/neutrophiles/caffe_models/model/'
deployfolder = '/home/saj/Downloads/caffe-master/examples/neutrophiles/caffe_models/deploy/'

modelfiles = sorted(glob.glob(modelfolder + '*.caffemodel'))
deployfiles = sorted(glob.glob(deployfolder + '*.prototxt'))
#%

#%

caffe.set_mode_gpu()

for netcount in range(np.shape(modelfiles)[0]):

    net = caffe.Net( deployfiles[netcount],modelfiles[netcount], caffe.TEST)
    print netcount,' ',net.params.keys()
#%%
#net = caffe.Net(caffe_root + 'examples/ecoli/ecolifile2deploy.prototxt',
#                caffe_root + 'examples/ecoli/file8bit2_iter_10000.caffemodel',
#                caffe.TEST)
#net = caffe.Net(caffe_root + 'examples/ecoli/neutro3classv3_deploy.prototxt',
#                caffe_root + 'examples/ecoli/neutro3clasv3_iter_10000.caffemodel',
#                caffe.TEST)
    imgfolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/sampimg/'
    outfolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/output/neuralnet_caffedirect/ver3/'
    inputimgfiles = sorted(glob.glob(imgfolder + '*.tif'))
#    inputimgfiles = inputimgfiles[0:2]
    
        
#    filecount = 0
    for infile in inputimgfiles:
#        st = time.time()
#orimg = caffe.io.load_image(inputimgfiles[1])
        orimg = caffe.io.load_image(infile)
        
        outimg = np.zeros((orimg.shape[0],orimg.shape[1],3))
        inimg = np.zeros((3,orimg.shape[0],orimg.shape[1]))
        inimg[0,:,:] = orimg[:,:,0]
        inimg[1,:,:] = orimg[:,:,0]
        inimg[2,:,:] = orimg[:,:,0]
        
        #ii = jj = 100
        #net.predict([orimg[ii-30:ii+30,jj-30:jj+30,:] ])
        
        
        
        outimg = np.float32(outimg)
        #%%
        #a = []
        step = 4
        for ii in np.arange(40,inimg.shape[1]-40,step):
            st = time.time()
            a = []
            count = ii
            for kk in range(step):
                for jj in np.arange(40,inimg.shape[2]-40,1):
                    a.append(inimg[:,count-31:count+31,jj-31:jj+31] )
                count += 1
            
            outimg[ii:ii+step, 40:orimg.shape[1]-40,:] = net.forward_all(data=np.array(a))['prob'].reshape((step,np.shape(a)[0]/step,3 ))            
            print ii, '  ', time.time()-st
        
    #    plt.imshow(outimg)
        savename = string.split(string.split(modelfiles[netcount],'/')[-1], '.')[0]
        savename = savename +'_'+ string.split(string.split(infile,'/')[-1], '.')[0]

        cv2.imwrite(outfolder+savename+'.tif',np.uint8(outimg*255))
#        print time.time()-st
        print savename,' classification done!'
#        filecount += 1
