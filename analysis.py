from util import *
import numpy as np
import options
import sys
import os
def get__errors_stats(filenames):
    '''get__errors_stats: get the mean and var error rate of the given files.
    The file should be produced by Convnet.
    Params:
        filenames: a string list contain the input files.'''
    l = []
    for f in filenames:
        dic = unpickle(f)
        import pdb
        pdb.set_trace()
        l += dic['model_state']['test_outputs'][-1][0]['logprob'][1]
    return l,np.mean(l),np.var(l)

if __name__ == '__main__':
    values,mean,var = get_average_error(sys.argv[1:])
    print 'error rates: ',values
    print 'mean error rate: ', mean
    print 'var error rate: ', var
