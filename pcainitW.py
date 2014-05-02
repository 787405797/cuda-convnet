# initialize weights using PCA

import numpy as np
import scipy.io as sio
from util import unpickle
import pylab as pl
import options
from math import ceil,sqrt

def pcainitW(name, idx, shape, model, params=None):
    ''' name: the layer name.
        idx: the weight matrix within that layer.
        shape: a 2-tuple of (rows, columns).
        dic: model state dictionary.
        params: string parameters specified in the definition.'''
    filter_path = "/home/pris/PCANet/Genki4k_filter.mat"
    dic = sio.loadmat(filter_path)
    filters = dic[params]
    row,col,num = filters.shape
    assert row*col == shape[0] and num == shape[1], "dimension not agree. Given:(%d,%d); required: (%d,%d)" % (row*col,num,shape[0],shape[1])
    filters = np.reshape(filters,shape)
    return filters

def showW(f):
    num = f.shape[-1]
    f = np.reshape(f,(sqrt(f.shape[0]),sqrt(f.shape[0]),num))
    row = 4
    col = ceil(num/4)
    pl.figure()
    for i in xrange(num):
        pl.subplot(row,col,i+1)
        pl.imshow(f[:,:,i],cmap=pl.cm.gray, interpolation='nearest')
        pl.title('f_%d' % (i+1))
    pl.show()

if __name__ == "__main__":
    name = 0
    idx = 0
    shape = (81,16)
    model = 0
    params = 'filter1'
    filter1 = pcainitW(name, idx, shape, model, params)
    showW(filter1)


