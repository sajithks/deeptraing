# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:55:25 2015

@author: saj
"""

#%%
a = cv2.imread('/home/saj/Documents/deep/deeptraing/data_neutrophils/draw/Aligned0060.png',-1)[:,:,0]

l,ncc= label(a==0,np.ones((3,3)))
large = 0
llab = 0
for ii in np.unique(l)[1:]:
    if(np.argwhere(l==ii).shape[0]>large):
        llab = ii
        large = np.argwhere(l==ii).shape[0]

cv2.imwrite('/home/saj/Documents/deep/deeptraing/data_neutrophils/labels/cell/Aligned0060.png',np.uint8((l*(l!=llab)>0)))

