from util import *
import numpy as np
import options
import sys
import os
import pylab as pl
from math import sqrt, ceil, floor
def get_errors_stats(filenames):
    '''get__errors_stats: get the mean and var error rate of the given files.
    The file should be produced by Convnet.
    Params:
        filenames: a string list contain the input files.'''
    l = []
    for f in filenames:
        dic = unpickle(f)
        l += [dic['model_state']['test_outputs'][-1][0]['logprob'][1]]
    return l,np.mean(l),np.var(l)

def show_error_rate(filename):
    dic = unpickle(filename)
    numbatches = len(dic['op'].options['train_batch_range'].value)
    train_outputs = dic['model_state']['train_outputs']
    test_outputs = dic['model_state']['test_outputs']
    train_errors = [o[0]['logprob'][1] for o in train_outputs]
    test_errors = [o[0]['logprob'][1] for o in test_outputs]
    numepoches = len(train_outputs) / numbatches
    testing_freq = dic['op'].options['testing_freq'].value

    train_errors = train_errors[:numepoches * numbatches]
    train_errors = np.reshape(train_errors,(numepoches,numbatches))
    train_errors = np.mean(train_errors,1)
    train_errors = list(train_errors.flatten())

    assert testing_freq % numbatches == 0, 'testing_freq should be n * numbatches' 
    test_errors = np.row_stack(test_errors)
    test_errors = np.tile(test_errors, (1, testing_freq / numbatches))
    test_errors = list(test_errors.flatten())
    test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
    test_errors = test_errors[:len(train_errors)]

    pl.figure(1)
    x = range(0, len(train_errors))
    pl.plot(x, train_errors, 'k-', label='Training set')
    pl.plot(x, test_errors, 'r-', label='Test set')
    pl.legend()
    ticklocs = range(1, numepoches+1)
    epoch_label_gran = int(ceil(numepoches / 20.)) # aim for about 20 labels
    epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
    ticklabels = map(lambda x: str(x[1]) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

    pl.xticks(ticklocs, ticklabels)
    pl.xlabel('Epoch')
#    pl.ylabel(self.show_cost)
    pl.title('logprob')
    pl.show()

if __name__ == '__main__':
    if sys.argv[1] == '-e':
        values,mean,var = get_errors_stats(sys.argv[2:])
        print 'error rates: ',values
        print 'mean error rate: ', mean
        print 'var error rate: ', var
    elif sys.argv[1] == '-s':
        show_error_rate(sys.argv[2])
    else:
        print 'analysis usage:'
        print '-e filaname1 filename2 ...'
        print '-s filename'
        exit(0)
