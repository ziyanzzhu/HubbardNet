#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:18:10 2023

@author: zoe
"""

import time 
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np
from tqdm import tqdm

from HubbardNet_gpu import *
import matrix_element as me

from os import path
import os
from copy import copy


plt.rcParams.update({'font.size': 20})
plt.rc('text',usetex=True)
#font = {'family':'serif','size':16}
font = {'family':'serif','size':25, 'serif': ['computer modern roman']}
plt.rc('font',**font)
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']


# Go to Edit -> Notebook Settings and select "GPU" from the hardware accelerator dropdown. 
# If this is on, GPU is enabled by default

use_gpu = False

# Check to see if gpu is available. If it is, use it else use the cpu
if torch.cuda.is_available() and use_gpu:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Using GPU.')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

    if not torch.cuda.is_available() and use_gpu: 
        use_gpu = False 
        print('GPU not available. Using CPU.')
    else: 
        print('Using CPU.')


model_list = []

M = 4
N_list = [M-1,M,M+1]

w = M
h = int(M/w)
pbc = True

for (n_idx, N) in enumerate(N_list): 
    O = N
    
    model = me.Bose_Hubbard(N, O, w=w, h=h, M=M, pbc=pbc)
    model_list.append(model)
    
    print("The size of the Hamiltonian is {}".format(model.tot_states))

t = 1 
U_max = 5
V = 0
U_list_all = np.arange(1, U_max+1, .5)
mu_list_all = np.linspace(0, 10, 5)
mu_list_all = np.array([0.])



## NN

S = 50 # number of sample of the MH sampler (not used)
init = 1 # the first state to sample in Metropolis Hastings (has nothing to do with the optimizer!) (not used)

# Network parameters
D_hid = 400 # the number ofneurons in the hidden layer
lr = 0.01 # learning rate 
epochs = 70000
loss_diff = 1e-7
grad_cut = 1e-6
check_point = 100 # print out the energy every X points
use_sampler = False # for now, only support ground state (not working anyway)

# Model parameters
U_train = np.ones(3)*2
t_train = 1.

U_train = np.array([4., 7, 10])
mu_train = np.zeros_like(U_train)

min_state = 0
max_state = 1

n_excited = max_state - 1

# paths to save and load weights 
fpath = os.getcwd()+'/weights/'

t0 = time.time()


# filepath for excited states
fname = fpath + "/weights_multi_N_M{}_Umax{}_Umin{}".format(M,np.max(U_train),np.min(U_train),D_hid)


def call_NN(lr, n_excited):
    if n_excited: 
        if n_excited == 1 :
            load_states_indv = [0]
        else:
            load_states_indv = range(1, n_excited)
        gs_flag = False
        es_flag = True

    else: # ground states
        load_states_indv = [0]
        gs_flag = True
        es_flag = False
    
    load_states = np.max(load_states_indv)  # total number of states being fixed 
  
    params = {'D_hid': D_hid, 
              'step_size': lr, 
              'max_iteration':epochs,
              'check_point': check_point,
              'loss_diff': loss_diff, 
              'steps': 1, # reset learning every N steps
              'loss_check_steps': 50, # check the local every N steps
              'grad_cut': grad_cut,  # stopping condition in the total gradient 
              'weight_init': False, 
              'zero_bias': False, 
              'gs_epochs': 1000, # the maximum number of steps to minimize the ground state
              'gs_flag': gs_flag, # ground state only
              'es_flag': es_flag,  # excited state only
              'regularization': True, 
              'load_states': load_states, # the number of states loaded 
              'load_states_indv': load_states_indv, 
              'rand_steps': 5000, 
              'load_weights_from_previous_state': False, # randomize the projection every N steps
              'use_gpu': use_gpu, 
              'weight_decay': 0.01,
              'perturb_amp': 0.0,
              'dropout': 0.0}
  
    print("Begin optimizing for state {}".format(n_excited))

    fc1, Loss_history, dot_history, all_E_list = train_NN(model_list, N_list, mu_train, U_train, t_train, V, S, params, fname, \
                                            use_sampler=use_sampler, init=init, loadweights=False,\
                                            fname_load=fname, n_excited=n_excited)

    return fc1, Loss_history, dot_history



fc1, Loss_history, dot_history = call_NN(lr, 0)

tf = time.time()
print("Training time = {} seconds.".format(tf-t0))

Loss_history = np.array(Loss_history)

#%% Plotting loss function

# fig,ax=plt.subplots(figsize=(7,5))
# ax.plot(Loss_history - np.min(Loss_history), '-', label='loss')
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Loss")
# # ax.set_title("U/t = {}".format(U_train))
# ax.set_yscale('log')


# # ax.plot(dot_history, '-', label='regularization')
# ax.set_xlabel("Iteration")
# # ax.set_ylabel("Dot product")
# # ax.set_title("U/t = {}".format(U_train))
# # ax.legend()
# plt.savefig('./figures/loss_M{}.pdf'.format(M,N,range(min_state, max_state)), format='pdf',bbox_inches='tight')
# plt.show()


# #%% Check energies
# E_all = np.zeros(( len(U_train), model.tot_states) )

# for (i, U) in enumerate(U_train):
#     _, _, H = model.H_Bose_Hubbard(t, U, V=V, mu=mu_train[i])
#     evals, evecs = np.linalg.eig(H)
#     idx = np.argsort(evals)
#     evecs = evecs[:,idx]
#     evals = evals[idx]
#     E_all[i] = evals
    

# U_test = np.linspace(1.5, 15, 20)
# mu_test = np.zeros_like(U_test)


# fig, ax = plt.subplots(figsize=(7,5))

# colors = ['r','b', 'k']
# for j,model in enumerate(model_list):
#     E_train, wf_gs = wf_e_calc(model_list[j], N_list[j], U_train, mu_train, t, V, 0, 0, fc1, use_gpu=True)
#     E_test, wf_gs_test = wf_e_calc(model_list[j], N_list[j], U_test, mu_test, t, V, 0, 0, fc1, use_gpu=True)

    
#     ax.scatter(U_train,E_train,s=50,c=colors[j], marker='s')
#     ax.scatter(U_test,E_test,s=50, c=colors[j], marker='x',lw=2)

#     arr = [] 
#     for (i, U) in enumerate(U_test):
#         _, _, H = model.H_Bose_Hubbard(t, U, V=V, mu=mu_train[0])
#         vals, vecs = np.linalg.eig(H)
#         vals_idx = np.argsort(vals)
#         vals = np.sort(vals)
#         arr.append(vals[0])
#     ax.plot(U_test,arr,colors[j],label='N={}'.format(N_list[j]))
# ax.set_xlim([min(U_test), max(U_test)])  
# ax.set_ylabel('Energy')
# ax.set_xlabel('$U$')
# plt.legend(frameon=False,prop={'size': 16})
# plt.savefig("./figures/energy_M{}.pdf".format(M), format='pdf',bbox_inches='tight')
# plt.show()


# #%% chemical potential & charge gap

# U_test = np.linspace(0.1, 20, 40)
# mu_test = np.zeros_like(U_test)
# E_gs_nn = np.zeros( (len(N_list), len(U_test)) )
# E_gs_train = np.zeros( (len(N_list), len(U_train)) )

# colors = ['r','b','k']
# for j,model in enumerate(model_list):
#     E_test, wf_gs_test = wf_e_calc(model_list[j], N_list[j], U_test, mu_test, t, V, 0, 0, fc1, use_gpu=True)
#     E_train, wf_gs = wf_e_calc(model_list[j], N_list[j], U_train, mu_train, t, V, 0, 0, fc1, use_gpu=True)
#     E_gs_nn[j,:] = E_test.T
#     E_gs_train[j,:] = E_train.T
    
# cgap = E_gs_nn[0] + E_gs_nn[2] - 2*E_gs_nn[1]
# mu_plus = E_gs_nn[2] - E_gs_nn[1]
# mu_minus = E_gs_nn[1] - E_gs_nn[0]

# cgap_train = E_gs_train[0] + E_gs_train[2] - 2*E_gs_train[1]
# mu_plus_train = E_gs_train[2] - E_gs_train[1]
# mu_minus_train = E_gs_train[1] - E_gs_train[0]

# #%%

# fig, ax=plt.subplots(figsize=(7,5))
# color = next(ax._get_lines.prop_cycler)['color']
# ax.plot(t/U_test, mu_plus/U_test, 'x', color=color)
# ax.plot(t/U_test, mu_minus/U_test, 'x', color=color)
# ax.plot(t/U_train, mu_plus_train/U_train, 's', color=color)
# ax.plot(t/U_train, mu_minus_train/U_train, 's', color=color)

# ax.set_xlabel('$t/U$')
# ax.set_ylabel('$\mu^\pm/U$')
# ax.set_ylim([-0.23, 1])
# ax.set_xlim([0, 0.5])
# plt.show()
# plt.savefig('./figures/NN_chemical_potential_M{}.pdf'.format(M), format='pdf',bbox_inches='tight')

# #%% Check wavefunctions

# n_list = np.zeros( (len(model_list),M) )

# fig, ax = plt.subplots(1, len(model_list), figsize=(15, 5))
# U_test = np.array( [5] ) 

# for model_idx,model in enumerate(model_list):
#     # rearrange states by symmetry 
#     all_states = np.zeros_like(model.all_states)
#     idx_list = np.zeros(model.tot_states,dtype=int)

#     for i in range(int(model.tot_states/2)): 
#         all_states[i] = model.all_states[i]
#         all_states[-i-1] = np.flip(model.all_states[i])
#         idx_list[i] = i
#         for j in range(model.tot_states):
#             if all(model.all_states[j]==all_states[-i-1]): 
#                 idx_list[-i-1] = j
#                 break

    
#     mu_test = np.zeros(len(U_test))
#     vals_all = np.zeros((model.tot_states, len(U_test)))
#     vals_idx_all = np.zeros_like(vals_all)
#     vecs_all = np.zeros((model.tot_states, model.tot_states, len(U_test)))

    
#     _, _, H = model.H_Bose_Hubbard(t, U, mu=mu_train[0])
#     vals, vecs = np.linalg.eig(H)
#     vals_idx_all[:,0] = np.argsort(vals)
#     vals_all[:,0] = vals
#     vecs_all[:,:,0] = vecs

#     E_test, wf_test = wf_e_calc(model, N_list[model_idx], U_test, mu_test, t, V, 0, 0, fc1, use_gpu=True)
    
#     s = 0 
#     if model_idx == 0:
#         ymax = 0.4
#     else: 
#         ymax = 0.3
#     ymin = 0
  
#     i=0
#     wf_here = wf_test[i*model.tot_states:(i+1)*model.tot_states,s].squeeze().double()

#     # calculate occupation numbers
#     for m_idx in range(M):
#         for state_idx in range(model.tot_states):
#             ci = wf_here[state_idx].cpu().detach().numpy()
#             ni = model.all_states[state_idx][m_idx]
#             n_list[model_idx,m_idx] += ci**2 * ni

#     wf_exact = np.abs(vecs_all[:,int(vals_idx_all[s, i]), i]).squeeze()
#     wf_nn = np.abs(wf_here.cpu().detach().numpy())
#     ax[model_idx].plot(wf_exact[idx_list], label="ED")
#     ax[model_idx].plot(wf_nn[idx_list], 'x--', label="HubbardNet")
    
#     ax[model_idx].set_title("N = {}".format(N_list[model_idx]))
#     ax[model_idx].set_ylim((ymin,ymax))

# ax[0].set_ylabel('$|\Psi_0 (\mathbf{n})|$')
# ax[0].set_xlabel('Component')
# ax[1].set_xlabel('Component')
# ax[1].legend(frameon=True, prop={'size': 16})
# plt.savefig(os.getcwd() + "/figures/wf_multi_N_M{}_gs.pdf".format(M,N_list[model_idx],s,n_excited), format='pdf',bbox_inches='tight')
# plt.show()




