# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:37:53 2018

@author: cfragada
"""
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorly as tl
import torch

# small tests to understand equivalence between tensordot and tmprod
# equivalent to a = reshape(0:11,[3,4]); in Matlab
a = np.arange(12.).reshape(3, 4, order='F')
# equivalent to b = reshape(0:23,[4,3,2]); in Matlab
b = np.arange(24.).reshape(4, 3, 2, order='F')
b.transpose([2, 0, 1])  # to see as in Matlab
# equivalent to a3 = reshape(0:5,[3,2]); in Matlab
a3 = np.arange(6.).reshape(3, 2, order='F')

c = np.tensordot(b, a, axes=([0], [1]))  # equivalent to c = tmprod(b,a,1);
c.transpose([2, 0, 1])  # to see as in Matlab

c = np.tensordot(a3, b, axes=([1], [2]))  # equivalent to c = tmprod(b,a,1);
c.transpose([2, 1, 0])  # to see as in Matlab

c = np.tensordot(b, a3, axes=([2], [1]))  # equivalent to c = tmprod(b,a,1);
c.transpose([2, 0, 1])  # to see as in Matlab


def plot_results():
    # Violinplots
    label = ['dot', 'tensordot', 'tensorly', 'dotT', 'tensordotT', 'tensorlyT']
    pos = [1, 2, 3, 5, 6, 7]
    plt.violinplot([time_dense_dot, time_tensordot, time_tensorly,
                    time_dense_dotT, time_tensordotT, time_tensorlyT],
                   pos, vert=False, showmeans=True,
                   showextrema=False, showmedians=True, bw_method='silverman')
    plt.xlim([0, 2*max(np.mean(time_tensordot), np.mean(time_tensorly),
                       np.mean(time_dense_dot))])
    plt.yticks(pos, label)

    # Histograms
#    n_bins = int(total_it/10)
#    alpha = 0.5
#    plt.hist(time_dense_dot, bins=n_bins, alpha=alpha, label='dot')
#    plt.hist(time_tensordot, bins=n_bins, alpha=alpha, label='tensordot')
#    plt.legend(loc='upper right')


"""
Estimating practical Relative Complexity of D (the fast dictionary)
"""
experimental = True
verbose = True
show_plots = True
pytorch = False
total_it = 100
cpd_rank = 3

N_vec = np.array([10, 10, 10])  # [6, 6, 3]
N = N_vec.prod()

K_vec = np.array([10, 10, 10])  # [12, 12, 6]
K = K_vec.prod()

N_samples = 1  # 2000

D_terms = []
D_terms.append(np.random.randn(N_vec[0], K_vec[0]))
D_terms.append(np.random.randn(N_vec[1], K_vec[1]))
D_terms.append(np.random.randn(N_vec[2], K_vec[2]))

if pytorch:
    tl.set_backend('pytorch')
    D_terms_tl = [torch.from_numpy(x) for x in D_terms]
else:
    tl.set_backend('numpy')
    D_terms_tl = D_terms

if experimental:
    if verbose:
        print("Estimating RC with %d runs" % (total_it))

    time_dense_dot, time_dense_dotT = [], []
    time_tensordot, time_tensordotT = [], []
    time_tensorly, time_tensorlyT = [], []

    D_dense = np.random.randn(N, K)
    for k in range(total_it):
        # Testing over different sparsity levels
        p = float(k+1)/total_it  

#        x = np.zeros((k, 1))
#        nz = max(int(p*k), 1)   # not really bernoulli because the number active features is deterministic here.
#        idx = k*np.random.rand(1, nz)
#        x[idx.astype(int)] = np.random.randn(nz, 1)

#        xT = np.zeros((N, 1))
#        nz = max(int(p*N), 1) # not really bernoulli because the number of active features is deterministic here.
#        idx = N*np.random.rand(1, nz)
#        xT[idx.astype(int)] = np.random.randn(nz, 1)

        # Dense data - results do not change
        # Only one sample
#        x = np.random.randn(K, 1)  # results do not change if x sparse
#        xT = np.random.randn(N, 1)
        # Multiple samples
        x = np.random.randn(K, N_samples)  # results do not change if x sparse
        xT = np.random.randn(N, N_samples)

        # ## Dense dictionary with np.dot ###
        tic = time.time()
        D_dense.dot(x)
        toc = time.time()
        time_dense_dot.append(toc-tic)  # store time results

        tic = time.time()
        D_dense.T.dot(xT)
        toc = time.time()
        time_dense_dotT.append(toc-tic)  # store time results

        # ## Fast Dictionary - tensordot ###
        # tic = time.time()
        # X = np.reshape(x, K_vec, order='F')
        X = np.reshape(x,  np.append(K_vec, N_samples), order='F')
        tic = time.time()
#            Y = np.zeros(N_vec);
#            for r in range(self.nkron):
#            for mode in X.ndim
        for r in range(cpd_rank):
            np.tensordot(X, D_terms[0], axes=([0], [1]))
            np.tensordot(X, D_terms[1], axes=([1], [1]))
            np.tensordot(X, D_terms[2], axes=([2], [1]))
#        y2 = np.reshape(X,N_vec, order='F')
        toc = time.time()
        time_tensordot.append(toc-tic)  # store time results

        # tic = time.time()
        # Xt = np.reshape(xT, N_vec, order='F')
        Xt = np.reshape(xT, np.append(N_vec, N_samples), order='F')
        tic = time.time()
#            Y = np.zeros(N_vec);
#            for r in range(self.nkron):
#            for mode in X.ndim
        for r in range(cpd_rank):
            np.tensordot(Xt, D_terms[0].T, axes=([0], [1]))
            np.tensordot(Xt, D_terms[1].T, axes=([1], [1]))
            np.tensordot(Xt, D_terms[2].T, axes=([2], [1]))
#        y2 = np.reshape(Xt,K_vec, order='F')
        toc = time.time()
        time_tensordotT.append(toc-tic)  # store time results

        # ## Fast Dictionary - tensorly ###
        # tic = time.time()
        # X = np.reshape(x, K_vec, order='F')
        X = np.reshape(x,  np.append(K_vec, N_samples), order='F')
        if pytorch:
            X = torch.from_numpy(X)
        tic = time.time()
        for r in range(cpd_rank):
            tl.tenalg.multi_mode_dot(X, D_terms_tl)  # modes=[0, 1, 2]
        toc = time.time()
        time_tensorly.append(toc-tic)  # store time results

        # tic = time.time()
        # Xt = np.reshape(xT, N_vec, order='F')
        Xt = np.reshape(xT, np.append(N_vec, N_samples), order='F')
        if pytorch:
            Xt = torch.from_numpy(Xt)
        tic = time.time()
        for r in range(cpd_rank):
            tl.tenalg.multi_mode_dot(Xt, [x.transpose(1, 0) for x in D_terms_tl])
        toc = time.time()
        time_tensorlyT.append(toc-tic)  # store time results

    time_dense_dot = np.array(time_dense_dot)
    time_dense_dotT = np.array(time_dense_dotT)
    time_tensordot = np.array(time_tensordot)
    time_tensordotT = np.array(time_tensordotT)
    time_tensorly = np.array(time_tensorly)
    time_tensorlyT = np.array(time_tensorlyT)

    RC_tensordot = np.mean(time_tensordot)/np.mean(time_dense_dot)
    RC_tensordotT = np.mean(time_tensordotT)/np.mean(time_dense_dotT)

    RC_tensorly = np.mean(time_tensorly)/np.mean(time_dense_dot)
    RC_tensorlyT = np.mean(time_tensorlyT)/np.mean(time_dense_dotT)

    if verbose:
        print("RC_tensordot = %1.3f" % (RC_tensordot))
        print("RC_tensordotT = %1.3f" % (RC_tensordotT))
        print("RC_tensorly = %1.3f" % (RC_tensorly))
        print("RC_tensorlyT = %1.3f" % (RC_tensorlyT))
    if show_plots:
        plot_results()

else:  # only theoretical RC
    print(">>>> Theoretical RC not implemented yet! <<<<")
