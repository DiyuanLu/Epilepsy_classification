import numpy as np
import pandas as pd
import os
import ipdb
import fnmatch
import matplotlib.pyplot as plt
from scipy.stats import zscore
import time

def find_files(directory, pattern='Data*.csv', withlabel=True):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if withlabel:
                #if 'Data_F' in filename:
                    #label = '1'
                #elif 'Data_N' in filename:
                    #label = '0'
                if 'base' in filename:
                    label = '0'
                elif 'tip' in filename:
                    label = '1'
                elif 'seizure' in filename:
                    label = '2'
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))
    #random.shuffle(files)   # randomly shuffle the files
    return files

def read_data(filename, header=None, ifnorm=True ):
    '''read data from .csv
    return:
        data: 2d array [seq_len, channel]'''

    data = pd.read_csv(filename, header=header, nrows=None)
    data = data.values   ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data)
        data = data_norm    ##data.shape (2048, 1)

    #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]

    return data


def ApEn(U, m=2, r=0.2):
    '''compute the approximate entropy
    Param:
        U: time series
        m: specifies the pattern length, suggested to be 1 or 2
        r: defines the threshold of similarity. seggested to be 0.1std ~ 0.25std'''

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))


def SampEn(U, m, r):


    def _maxdist(x_i, x_j):

        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        return result


    def _phi(m):

        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]

        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]

        return sum(C)


    N = len(U)

    

    return -np.log(_phi(m+1) / _phi(m))


def renyientropy(px,alpha=1,logbase=2,measure='R'):
    """
    Renyi's generalized entropy
    Parameters
    ----------
    px : array-like
        Discrete probability distribution of random variable X.  Note that
        px is assumed to be a proper probability distribution.
    logbase : int or np.e, optional
        Default is 2 (bits)
    alpha : float or inf
        The order of the entropy.  The default is 1, which in the limit
        is just Shannon's entropy.  2 is Renyi (Collision) entropy.  If
        the string "inf" or numpy.inf is specified the min-entropy is returned.
    measure : str, optional
        The type of entropy measure desired.  'R' returns Renyi entropy
        measure.  'T' returns the Tsallis entropy measure.
    Returns
    -------
    1/(1-alpha)*log(sum(px**alpha))
    In the limit as alpha -> 1, Shannon's entropy is returned.
    In the limit as alpha -> inf, min-entropy is returned.
    """
#TODO:finish returns
#TODO:add checks for measure
    if not _isproperdist(px):
        raise ValueError("px is not a proper probability distribution")
    alpha = float(alpha)
    if alpha == 1:
        genent = shannonentropy(px)
        if logbase != 2:
            return logbasechange(2, logbase) * genent
        return genent
    elif 'inf' in string(alpha).lower() or alpha == np.inf:
        return -np.log(np.max(px))

    # gets here if alpha != (1 or inf)
    px = px**alpha
    genent = np.log(px.sum())
    if logbase == 2:
        return 1/(1-alpha) * genent
    else:
        return 1/(1-alpha) * logbasechange(2, logbase) * genent




data_dir = 'data/test_files'
save_name = "data/test_files/"
'''slide and seg'''

#seq_len = 10240
#width = 2

files_wlabel = find_files(data_dir, pattern='*.csv', withlabel=False)
#num_files = len(files_wlabel)

#datas = np.zeros((100, seq_len, width))
#labels = np.array(files_wlabel)[:, 1].astype(np.int)
#filenames = np.array(files_wlabel)[:, 0].astype(np.str)
'''cluster'''
#files_wlabel = ['data/test_files/sep09_anno_8_test-long.csv']
for ind, filename in enumerate(files_wlabel):
    print(filename)
    data = read_data(filename, header=None, ifnorm=True)
    t1 = time.time()
    ae_data1 = ApEn(data[:, 0])   ###('time', 563.8928680419922)
    ae_data2 = ApEn(data[:, 1])
    print("time", time.time()-t1)
    ipdb.set_trace()
    #np.savetxt(save_name, aug_data, header="datax,datay,corrx,corry", delimiter=',', comments='')
