# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:05:03 2015

@author: root
"""

#%%
neuralfolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/neuralnet_caffedirect/ver4/'
nntestfiles = sorted(glob.glob(neuralfolder + '*.png'))
outfolder1 = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/neuralnet_caffedirect/seg/'
outfolder2 = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/randomforest/seg/'

for ii in nntestfiles:
    savname = string.split(ii,'/')[-1]
    img = cv2.imread(ii,-1)
    outimg = np.argmax(img,2)
    cv2.imwrite(outfolder1+savname,cb.normalizeImage(outimg))

#random forest
rffolder = '/Users/sajithks/Documents/deeptraing/data_neutrophils/output/randomforest/ver4/'
rftrainfiles = sorted(glob.glob(rffolder + '*.png'))

for ii in rftrainfiles:
    savname = string.split(ii,'/')[-1]
    img = cv2.imread(ii,-1)
    outimg = np.argmax(img,2)
    cv2.imwrite(outfolder2+savname,cb.normalizeImage(outimg))
