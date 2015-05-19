# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:16:23 2015

@author: saj
"""




import scipy as sp
import numpy as np
caffe_root = '/home/saj/Downloads/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import pickle


#%% 
#print 'network training done'
###############################################################################


caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'examples/neutrophiles/caffe_models/deploy/16_16_16.prototxt',
                caffe_root + 'examples/neutrophiles/caffe_models/model/16_16_16_iter_10000.caffemodel',
                caffe.TEST)
#net = caffe.Net(caffe_root + 'examples/ecoli/neutro3classv3_deploy.prototxt',
#                caffe_root + 'examples/ecoli/neutro3clasv3_iter_10000.caffemodel',
#                caffe.TEST)


outfolder = '/home/saj/Documents/deep/deeptraing/data_neutrophils/caffe_net/ver3/'

#%%
caffenet = {}
for ii in net.params.keys():
    caffenet[ii+'_filt'] = net.params[ii][0].data
    caffenet[ii+'_bias'] = net.params[ii][1].data
    


pickle.dump( caffenet, open( outfolder +'neutro_conv_16_16_16.p', "wb" ) )


#%%
#caffenet = pickle.load( open( outfolder+'neutronet.p', "rb" ) )



