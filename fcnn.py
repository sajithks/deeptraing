# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:54:27 2015

@author: root
"""

#import pickle

#import scipy as sp
from scipy.signal import convolve
import time
#import sys
#import os
#sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf')
#sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/ecoli_det_seg')
#import createbacteria  as cb
import numpy as np

import itertools 
#print 'libraries loaded'

def fastConvolve(inimg, filt):
    outnums = np.shape(filt)[0]
    
    convrow = inimg.shape[1] - 2*(filt.shape[-2]/2)
    convcol = inimg.shape[2] - 2*(filt.shape[-1]/2)
    
    convout = np.zeros((outnums, convrow, convcol))
    for ii in range(outnums):
        filterval = filt[ii]
        convout[ii,:,:] = convolve(inimg, filterval,'valid')[0]
    return convout

def checkEven(inimg):
    if(np.mod(inimg.shape[1],2 )==1 ):
        inimg = inimg[:,0:inimg.shape[1]-1,:]
    if(np.mod(inimg.shape[2], 2 )==1 ):
        inimg = inimg[:,:,0:inimg.shape[2]-1]
    return(inimg)

def simplePool(inimg):
    tempmax = []
    for ii in range(2):
        for jj in range(2):
            tempmax.append(inimg[:, ii:inimg.shape[-2]:2, jj:inimg.shape[-1]:2])
    return(np.array(np.max(tempmax,0)))
    
    
def fastMaxPool(inimg):
    '''
    max pool image
    
    '''
    inimg = checkEven(inimg)
    pooloutimg = []
    for rolx in range(2):
        for roly in range(2):
            tempimg = np.roll(inimg,-rolx, 1)
            tempimg = np.roll(tempimg, -roly, 2)
            
            pooloutimg.append(simplePool(tempimg))
    
    return(np.array(pooloutimg))

def flipFilter(filt):
    '''
    flip filter for correlation operation in caffe
    '''
    nfilt = np.zeros_like(filt)
    for ii in range(filt.shape[0]):
        for jj in range(filt.shape[1]):
            nfilt[ii,jj,:,:] = np.flipud(np.fliplr(filt[ii,jj,:,:]))
    return(nfilt)

def distributMatrix(inimg, subimage, stride):
    
    indexmatrix = np.zeros( (inimg.shape[-2],inimg.shape[-1]))
    for ii in range(stride):
        indexmatrix[ii::stride,ii::stride] = 1
    if(inimg.shape[0]==1):
        inimg[indexmatrix] = subimage.reshape(subimage.shape[0]*subimage.shape[1])
    else:
        inimg[:,indexmatrix] = subimage.reshape(subimage.shape[0],subimage.shape[1]*subimage.shape[2])

    return(inimg)

#%%
def extractFcnnFeature(orimg, caffenet):
    '''
    extractFcnnFeature(orimg, caffenet) --> outfeatimg
    extract fcnn feature for the given image for fixed 3 convolution layer
    network
    '''
    orimg = np.float32(orimg)
    orimg = orimg - orimg.min()
    orimg = 255*orimg/orimg.max()
    
    outimg = np.zeros((orimg.shape[0],orimg.shape[1],3))
    
    inimg = np.zeros((3,orimg.shape[0],orimg.shape[1]))
    inimg[0,:,:] = orimg
    inimg[1,:,:] = orimg
    inimg[2,:,:] = orimg
    
    
#    start = time.time()
#    print 'extracting features ...'
    #st = time.time()
    
    #pval = ['conv3','conv3_level2','conv3_level3']
    
    #filtname = 'conv3'
    filt = caffenet['conv3_filt']
    bias = caffenet['conv3_bias']
        
    a1 = fastConvolve(inimg, flipFilter(filt) )
    
    for ii in range(filt.shape[0]):
        a1[ii] = a1[ii] +bias[ii]
        
    
    maxoutl1 = fastMaxPool(a1)
    
    #%level 2
    
    #filtname = 'conv3_level2'
    filt = caffenet['conv3_level2_filt']
    bias = caffenet['conv3_level2_bias']
    
    maxoutl2 = []
    for ii in range(np.shape(maxoutl1)[0]):
        a2 = fastConvolve(maxoutl1[ii], flipFilter(filt))
        for fil in range(filt.shape[0]):
            a2[fil] = a2[fil] + bias[fil]    
        maxoutl2.append(fastMaxPool( a2 ) )
    
    #%level 3
    
    #filtname = 'conv3_level3'
    filt = caffenet['conv3_level3_filt']
    bias = caffenet['conv3_level3_bias']
    
    maxoutl3 = []
    coutl3 = []
    for ii in range(np.shape(maxoutl2)[0] ):
        templ3 = []
        for jj in range(np.shape(maxoutl2)[1]):
            a3 = fastConvolve(maxoutl2[ii][jj,:,:,:], flipFilter(filt) )
            for fil in range(filt.shape[0]):
                a3[fil] = a3[fil] + bias[fil]                    
            templ3.append( fastMaxPool(a3))
        maxoutl3.append(templ3)   
    
    #%
        
    maxdim = np.shape(maxoutl3)
    
    maxoutl3arr = np.zeros((64,2, maxdim[-2], maxdim[-1]))  
    mxcount = 0   
    
    rowindex = np.arange(0, maxdim[-2]*8, 1)
    colindex = np.arange(0, maxdim[-1]*8, 1)
    
    outfeatimg = np.zeros((maxdim[-2]*8,maxdim[-1]*8,maxdim[-3]))
    
    for ii in range(np.shape(maxoutl3)[0]):
        for jj in range(np.shape(maxoutl3)[1]):
            for kk in range(np.shape(maxoutl3)[2]):
                l1row = np.int32(np.binary_repr(ii,2)[0])
                l1col = np.int32(np.binary_repr(ii,2)[1])
    
                l2row = np.int32(np.binary_repr(jj,2)[0])
                l2col = np.int32(np.binary_repr(jj,2)[1])
    
                l3row = np.int32(np.binary_repr(kk,2)[0])
                l3col = np.int32(np.binary_repr(kk,2)[1])
    
                r = rowindex[l1row::2][l2row::2][l3row::2]
                c = colindex[l1col::2][l2col::2][l3col::2]     
                arrayindex = np.array(list(itertools.product(r,c))) 
                temparr = maxoutl3[ii][jj][kk][ :]
                tem = []            
                for cc in range(temparr.shape[0]):
                    tt = temparr[cc]
                    tem.append(tt.reshape(tt.shape[0]*tt.shape[1],order='C'))
                tem = np.array(tem).T
                
                outfeatimg[arrayindex.T[0],arrayindex.T[1],:] = tem
    
    return(outfeatimg, maxoutl3)

###############################################################################
def classifyFcnnFeature(orimg, caffenet):
    '''
    classifyFcnnFeature(orimg, caffenet) --> outimg
    extract and classify fcnn feature for the given image for fixed 3 convolution layer
    network
    '''
    orimg = np.float32(orimg)
    orimg = orimg - orimg.min()
    orimg = 255*orimg/orimg.max()
    
    outimg = np.zeros((orimg.shape[0],orimg.shape[1],3))
    
    inimg = np.zeros((3,orimg.shape[0],orimg.shape[1]))
    inimg[0,:,:] = orimg
    inimg[1,:,:] = orimg
    inimg[2,:,:] = orimg
    
    
#    start = time.time()
#    print 'extracting features ...'
    #st = time.time()
    
    #pval = ['conv3','conv3_level2','conv3_level3']
    
    #filtname = 'conv3'
    filt = caffenet['conv3_filt']
    bias = caffenet['conv3_bias']
        
    a1 = fastConvolve(inimg, flipFilter(filt) )
    
    for ii in range(filt.shape[0]):
        a1[ii] = a1[ii] +bias[ii]
        
    
    
    maxoutl1 = fastMaxPool(a1)
    
    #%level 2
    
    #filtname = 'conv3_level2'
    filt = caffenet['conv3_level2_filt']
    bias = caffenet['conv3_level2_bias']
    
    maxoutl2 = []
    for ii in range(np.shape(maxoutl1)[0]):
        a2 = fastConvolve(maxoutl1[ii], flipFilter(filt))
        for fil in range(filt.shape[0]):
            a2[fil] = a2[fil] + bias[fil]    
        maxoutl2.append(fastMaxPool( a2 ) )
    
    #%level 3
    
    #filtname = 'conv3_level3'
    filt = caffenet['conv3_level3_filt']
    bias = caffenet['conv3_level3_bias']
    
    maxoutl3 = []
    coutl3 = []
    for ii in range(np.shape(maxoutl2)[0] ):
        templ3 = []
        for jj in range(np.shape(maxoutl2)[1]):
            a3 = fastConvolve(maxoutl2[ii][jj,:,:,:], flipFilter(filt) )
            for fil in range(filt.shape[0]):
                a3[fil] = a3[fil] + bias[fil]                    
            templ3.append( fastMaxPool(a3))
        maxoutl3.append(templ3)   
    
    #%
        
    maxdim = np.shape(maxoutl3)
    
    maxoutl3arr = np.zeros((64,2, maxdim[-2], maxdim[-1]))  
    mxcount = 0   
    
    rowindex = np.arange(0, maxdim[-2]*8, 1)
    colindex = np.arange(0, maxdim[-1]*8, 1)
    
    outfeatimg = np.zeros((maxdim[-2]*8,maxdim[-1]*8,maxdim[-3]))
    
    for ii in range(np.shape(maxoutl3)[0]):
        for jj in range(np.shape(maxoutl3)[1]):
            for kk in range(np.shape(maxoutl3)[2]):
                l1row = np.int32(np.binary_repr(ii,2)[0])
                l1col = np.int32(np.binary_repr(ii,2)[1])
    
                l2row = np.int32(np.binary_repr(jj,2)[0])
                l2col = np.int32(np.binary_repr(jj,2)[1])
    
                l3row = np.int32(np.binary_repr(kk,2)[0])
                l3col = np.int32(np.binary_repr(kk,2)[1])
    
                r = rowindex[l1row::2][l2row::2][l3row::2]
                c = colindex[l1col::2][l2col::2][l3col::2]     
                arrayindex = np.array(list(itertools.product(r,c))) 
                temparr = maxoutl3[ii][jj][kk][ :]
                tem = []            
                for cc in range(temparr.shape[0]):
                    tt = temparr[cc]
                    tem.append(tt.reshape(tt.shape[0]*tt.shape[1],order='C'))
                tem = np.array(tem).T
                
                outfeatimg[arrayindex.T[0],arrayindex.T[1],:] = tem
    
    ip1_filt = caffenet['ip1_filt']
    ip1_bias = caffenet['ip1_bias']
    
    ip2_filt = caffenet['ip2_filt']
    ip2_bias = caffenet['ip2_bias']
    
    ip3_filt = caffenet['ip3_filt']
    ip3_bias = caffenet['ip3_bias']
    #%
    dim = np.shape(maxoutl3)
    
    outimg = []
    classout = np.zeros((dim[0], dim[1], dim[2], ip3_filt.shape[0], dim[4]-6, dim [5]-6))
    
    for ii in range(np.shape(maxoutl3)[0]):
        print ii,jj,kk

        for jj in range(np.shape(maxoutl3)[1]):
            for kk in range(np.shape(maxoutl3)[2]):
                featimg = maxoutl3[ii][jj][kk]
                rindex = 0
                feat1d = []
                for row in np.arange(3, np.shape(featimg)[1]-3, 1):
                    cindex = 0  
                    for col in np.arange(3,np.shape(featimg)[2]-3, 1):
                        feat1d.append(featimg[:,row-3:row+3,col-3:col+3].reshape((featimg.shape[0]*36),order='C'))
                        
                inner1 = (np.inner(feat1d,ip1_filt)+ip1_bias)#ip1
#                inner1 = inner1*(inner1>0)#relu
                inner2 = (np.inner(inner1,ip2_filt)+ip2_bias)
#                inner2 = inner2*(inner2>0)
                inner3 = (np.inner(inner2,ip3_filt)+ip3_bias)
#                        a = (np.inner(feat1d,ip1_filt)+ip1_bias)#/(np.sum(np.inner(feat1d,ipfilt) +bias))            
                    #softmax    
                classout[ii, jj, kk, 0, :, :] = np.divide(np.exp(inner3)[:, 0], np.sum(np.exp(inner3), 1) ).reshape((dim[4]-6, dim [5]-6))
                classout[ii, jj, kk, 1, :, :] = np.divide(np.exp(inner3)[:, 1], np.sum(np.exp(inner3), 1) ).reshape((dim[4]-6, dim [5]-6))
                classout[ii, jj, kk, 2, :, :] = np.divide(np.exp(inner3)[:, 2], np.sum(np.exp(inner3), 1) ).reshape((dim[4]-6, dim [5]-6))
                    
#                        cindex +=1
#                rindex += 1
                    
    
    
    
    
    #%%
                    
    level_mult = 8
    dimclass = np.shape(classout)
    
    outimg = np.zeros((dimclass[-2] * level_mult,dimclass[-1]* level_mult, dimclass[-3]))                
    rowindex = np.arange(0, outimg.shape[0], 1)
    colindex = np.arange(0, outimg.shape[1], 1)
    
    for ii in range(dimclass[2]):
        l1row = np.int32(np.binary_repr(ii,2)[0])
        l1col = np.int32(np.binary_repr(ii,2)[1])
        for jj in range(dimclass[1]):
            l2row = np.int32(np.binary_repr(jj,2)[0])
            l2col = np.int32(np.binary_repr(jj,2)[1])
            for kk in range(dimclass[0]):
                l3row = np.int32(np.binary_repr(kk,2)[0])
                l3col = np.int32(np.binary_repr(kk,2)[1])
                r = rowindex[l1row::2][l2row::2][l3row::2]
                c = colindex[l1col::2][l2col::2][l3col::2]     
                arrayindex = np.array(list(itertools.product(r,c))) 
                a = classout[ii,jj,kk,:,:,:]            
                outimg[arrayindex.T[0],arrayindex.T[1]] = np.array([a[0,:,:].reshape(dimclass[-2]*dimclass[-1]),a[1,:,:].reshape(dimclass[-2]*dimclass[-1]),a[2,:,:].reshape(dimclass[-2]*dimclass[-1])]).T

    return(outimg)
