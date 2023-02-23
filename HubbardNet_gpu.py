#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:43:29 2021

@author: zoe

for testing multiple excited states
"""

import numpy as np
from scipy import stats
# import matrix_element as me 
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix 
# import netket as nk
import torch 
import torch.optim as optim
from torch.autograd import grad
import copy
import time
from os import path


# find nullspace or the orthogonal space (this requires python 3.9 and pytorch 1.8.0+)
def my_nullspace(At, rcond=None):
    '''
    If the input is a vector, its dimension needs to be [1, N] 
    '''
    ut, st, vht = torch.linalg.svd(At, full_matrices=True)      
    Mt, Nt = ut.shape[0], vht.shape[1] 
    if rcond is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcondt
    numt= torch.sum(st > tolt, dtype=int)
    nullspace = vht[numt:,:].T.conj()
    nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
    return nullspace


def project(B): 
    ''' 
    If the input is a vector, its dimension needs to be [N, 1]
    Note: output from my_nullspace has the right dimension
    '''
    return B.matmul( torch.linalg.inv( (B.T.matmul(B)).T ).matmul( B.T) )


def normalize(u): 
    return u / torch.sqrt( u.dot(u))

def proj_vec(u, v): 
    return (u.dot(v) / u.dot(u)) * u 


def Gram_Schmidt( A, u1, seed=0, use_gpu=False):
    '''
    Perform Gram Schmidt on A such that every pair of A[:,i] and A[:, j] are orthogonal for i != j  
    ground state given as u1
    '''
    if use_gpu: 
        idx = torch.randperm( A.shape[1], generator = torch.cuda.manual_seed(seed))
    else: 
        idx = torch.randperm( A.shape[1], generator = torch.manual_seed(seed))
    
    if len(u1.shape) == 1:
        u1 = u1.reshape( (len(u1),1) )
    
    B = torch.cat((u1.clone(), A[:,idx].clone()), dim=1)
    A_ortho = torch.zeros_like(A)
    
    for i in range(u1.shape[1] ,A_ortho.shape[1]+u1.shape[1]): 
       
        u = torch.cat( (u1.clone(), A_ortho[:,0:i-u1.shape[1]].clone()), dim=1)
        v = B[:,i].clone() # the vector to project on
        
        tmp = v.clone().squeeze()
        for j in range(u.shape[1]): 
           
            uj = u[:,j].clone() # need to avoid in-place operations
            
            tmp = tmp - proj_vec(uj.squeeze(), v.squeeze())
            
        A_ortho[:,i-u1.shape[1]] = normalize(tmp)
        
    return A_ortho 


def Gram_Schmidt2( A ):
    '''
    Perform Gram Schmidt on A such that every pair of A[:,i] and A[:, j] are orthogonal for i != j  
    '''
    
    A_ortho = torch.zeros_like(A)

    for i in range(A_ortho.shape[1]): 
       
        u = A_ortho[:,0:i].clone()
        v = A[:,i].clone()
        
        tmp = v.squeeze()
        if i == 0: 
            A_ortho[:,i] = normalize(tmp)
        else: 
            for j in range(u.shape[1]): 
                
                uj = u[:,j].clone() # need to avoid in-place operations
                tmp = tmp - proj_vec(uj.squeeze(), v.squeeze())

            A_ortho[:,i] = normalize(tmp)
        
    return A_ortho 


def energy(wf_all, wf_gs, H_list, tot_states_list, mu_list, U_list, t, N_list, V, n, \
           regularization=False, seed=0, use_gpu=False):
    '''
    Need to test for multiple N's and U's

    Parameters
    ----------
    wf_all : excited states wavefunction
    wf_gs : ground state wavefunction
    model_list : list of model 
    mu_list : list of mu (chemical potential)
    U_list : list of Hubbard U parameter
    t : list of Hubbard t parameters
    N_list : list of N (same size as model_list)
    V : on-site potential
    n : the number of excited states
    regularization : whether regularization term is added (to encourage to be far from the ground state)

    Returns
    -------
    TYPE
        Mean average
    reg : regularization parameter

    '''
    
    if len(wf_gs.shape) == 1:
        wf_gs = wf_gs.reshape((len(wf_gs),1))
    E = torch.zeros(len(U_list)*len(N_list)*n)
    E_gs = torch.zeros(len(U_list)*len(N_list))
    idx = 0
    idx2 = 0 
    # n_state_list = [x for x in tot_sttes_list]


            
    for (j,N) in enumerate(N_list): 
     
        n_states = tot_states_list[j]
        
        wf = wf_all[ len(U_list)*int(np.sum(tot_states_list[0:j])): len(U_list)*int(np.sum(tot_states_list[0:j]) + len(U_list)*n_states), :]
        
    
        for (i,U) in enumerate(U_list):
            
            H = H_list[idx2]
            wf_gs_here =  wf_gs[i*n_states:(i+1)*n_states,:]
            
            # wf1_here = wf[i*model.tot_states:(i+1)*model.tot_states, 0]
            # wf_proj = wf_single_proj(wf_gs_here, wf1_here) # wf_proj is orthogonal to wf_gs_here/ ground state wavefunction
            # wf_here = torch.cat((wf_proj.reshape((len(wf_proj),1)), wf_here[:,1::]), 1)
    
            wf_here = wf[i*n_states:(i+1)*n_states, :]
            
            wf_ortho = Gram_Schmidt(wf_here, wf_gs_here, seed, use_gpu=use_gpu)
            
           
            for s in range(n):
                H = torch.tensor(H).double()
                wf_here = wf_ortho[:, s].squeeze()
                E_tmp = torch.mv(H, wf_here)
                E_tmp = torch.dot(E_tmp, wf_here) 
                E[idx] = E_tmp/torch.sqrt(torch.sum(wf_here.pow(2)))
                idx += 1
            

            # for s in range(wf_ortho.shape[1]):
            #     for s1 in range(wf_gs_here.shape[1]): 
                  
            #         dot = wf_gs_here[:,s1].dot(wf_ortho[:,s])
            #         print(dot)
            #         # if dot > 0.01:
            #         #     print(s, s1)


            E_tmp = torch.mv(H, wf_gs_here[:,0])
            E_tmp = torch.dot(E_tmp, wf_gs_here[:,0])
            E_gs[idx2] = E_tmp/torch.sqrt(torch.sum(wf_gs_here[:,0].pow(2)))
            
            # check the energies of previous states
            # for x in range(wf_gs_here.shape[1]): 
            #     E_tmp = torch.mv(H, wf_gs_here[:,x])
            #     E_tmp = torch.dot(E_tmp, wf_gs_here[:,x])
            #     E_tmp = E_tmp/torch.sqrt(torch.sum(wf_gs_here[:,x].pow(2)))
            #     print(E_tmp)
                
            idx2 += 1
    
    
    # if orthonormal, (v1 v2 ... vp) . (v1 v2 ... vp)^T = eye(p)
    # reg = wf_all.T.matmul(wf_all)
    # reg -= torch.eye(n)
    # reg = torch.abs(reg[:]).pow(2).sum()
    # regularization, making sure excited states energies are as close to the ground state energy as possible 
    
    if not regularization: 
        reg = torch.tensor([0.0])
    else: 
        reg = torch.exp( -torch.abs(E - E_gs) )
    
    

    return E.mean()-reg.mean(), reg, E
    

# impose the constraint from the eigenvalue equation 
def constraint(wf_all, wf_gs, H_list, tot_states_list, mu_test_list, U_list, t, N_list, V, n, \
           regularization=False, seed=0, use_gpu=False):
    '''

    '''
    if len(wf_gs.shape) == 1:
        wf_gs = wf_gs.reshape((len(wf_gs),1))
    E = torch.zeros(len(U_list)*len(N_list)*n)
    E_gs = torch.zeros(len(U_list)*len(N_list))
    constraint = torch.zeros_like(E)
    idx = 0
    idx2 = 0 
    # n_state_list = [x for x in tot_sttes_list]
            
    for (j,N) in enumerate(N_list): 
     
        n_states = tot_states_list[j]
        
        wf = wf_all[ len(U_list)*int(np.sum(tot_states_list[0:j])): len(U_list)*int(np.sum(tot_states_list[0:j]) + len(U_list)*n_states), :]
        
    
        for (i,U) in enumerate(U_list):
            
            H = H_list[idx2]
            wf_gs_here =  wf_gs[i*n_states:(i+1)*n_states,:]
            wf_here = wf[i*n_states:(i+1)*n_states, :]
            wf_ortho = Gram_Schmidt(wf_here, wf_gs_here, seed, use_gpu=use_gpu)
            
            for s in range(n):
                H = torch.tensor(H).double()
                wf_here = wf_ortho[:, s].squeeze()
                term1= torch.mv(H, wf_here)
                E_tmp = torch.dot(term1, wf_here) 
                E[idx] = E_tmp/torch.sqrt(torch.sum(wf_here.pow(2)))
                
                term2 = E[idx]*wf_here 
                constraint[idx] = torch.sum(term1-term2)
                idx += 1
                
    return constraint
    


def energy_1state(wf_all, H_list, n_state_list, mu_list, U_list, t, N_list, V):
    
    E = torch.zeros(len(U_list)*len(N_list))
    idx = 0

    for (j,N) in enumerate(N_list):
        
        n_states = n_state_list[j]
       
        wf = wf_all[ len(U_list)*int(np.sum(n_state_list[0:j])): int(np.sum(n_state_list[0:j])*len(U_list) + len(U_list)*n_states)]  # wavefunction for the given N
   
        for (i,U) in enumerate(U_list):
            
            H = H_list[idx]
            
            wf_here = wf[i*n_states:(i+1)*n_states]

            wf_here = wf_here.squeeze()
            E_tmp = torch.mv(H, wf_here)
            E_tmp = torch.dot(E_tmp, wf_here) 
            E[idx] = E_tmp/torch.sqrt(torch.sum(wf_here.pow(2)))
            idx += 1
    
    return E.mean()

def wf_proj( wf_all, model_list, U_list, N_list, use_gpu=False): 
     
    n_state_list = [model.tot_states for model in model_list]
    wf_re = torch.zeros_like(wf_all)
    
    for (j,N) in enumerate(N_list): 
        model = model_list[j]
        n_states = model.tot_states
        
        wf = wf_all[ len(U_list)*int(np.sum(n_state_list[0:j])): len(U_list)*int(np.sum(n_state_list[0:j]) + len(U_list)*n_states), :]
      
        wf_ortho = torch.zeros_like(wf)
        for (i,U) in enumerate(U_list):
             wf_ortho_tmp = Gram_Schmidt(wf[i*model.tot_states:(i+1)*model.tot_states, :], use_gpu=use_gpu)
             wf_ortho[i*model.tot_states:(i+1)*model.tot_states, :] = wf_ortho_tmp
        
        wf_re[len(U_list)*int(np.sum(n_state_list[0:j])): len(U_list)*int(np.sum(n_state_list[0:j]) + len(U_list)*n_states), :] = wf_ortho
                
    return wf_re


def wf_single_proj( wf1, wf2 ): 
    ''' 
    project wf2 onto the orthogonal plane of wf1
    '''
    
    # find nullspace of wf1 
    # shape wf1 into the proper shape
    wf_shape = wf1.shape
    if len(wf_shape) == 1:
        wf1 = wf1.reshape( (1, len(wf1)) )
    else: 
        wf1 = wf1.reshape( (min(wf_shape), max(wf_shape) ) )
        
    # print(wf1.shape)
        
    wf1_ortho = my_nullspace(wf1)
            
    # project the excited states wavefunction onto this space 
    # define projection matrix 
    P = project( wf1_ortho )
    # apply projection matrix on the excited state wavefunction
    wf_proj = torch.mv(P, wf2.squeeze())
    
    return wf_proj



# define the network 
class feedforward(torch.nn.Module):
    def __init__(self, M, D_hid=10,dropout=0.0):
        # Input: number of sites & the number of neurons in the hidden layers 
        '''
        Input parameters: M = number of sites, D_hid: number of neurons per layer 
        '''
        super(feedforward,self).__init__()

        # Define the Activation 
        self.actF = torch.nn.Tanh()
        # self.actF = torch.nn.LeakyReLU()
        
        # define layers
        # nn.Linear applies linear transformation to the incoming data y = x A^T + b
        self.Lin_1   = torch.nn.Linear(M+4, D_hid)
        self.Lin_2   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_3   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_4   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_5   = torch.nn.Linear(D_hid, D_hid)
        if dropout > 0: 
            self.dropout = torch.nn.Dropout(p=dropout)
        self.Lin_out = torch.nn.Linear(D_hid, 1) 
    
    def forward(self,states,UtN_list):
        # provide inputs give outputs
        # backward: based-on weights
                
        # layer 1
        l = self.Lin_1(torch.cat((states,UtN_list),1))
        l = self.Lin_2(l);    h = self.actF(l)
        l = self.Lin_3(h);    h = self.actF(l)
        l = self.Lin_4(h);    h = self.actF(l)
        l = self.Lin_5(h);    h = self.actF(l)
      
        # output layer, linear for regression
        netOut = self.Lin_out(h) 
        return netOut


# weight initialization
def weight_init(m, dist, mean=None, std=None, zero_bias=True): 
    if torch.jit.isinstance(m, torch.nn.Module):
        
        if dist == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.Lin_1.weight.data)            
            
        elif dist == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.Lin_1.weight.data)
            torch.nn.init.xavier_uniform_(m.Lin_2.weight.data)
            torch.nn.init.xavier_uniform_(m.Lin_3.weight.data)
            torch.nn.init.xavier_uniform_(m.Lin_4.weight.data)
            torch.nn.init.xavier_uniform_(m.Lin_5.weight.data)
            
        elif dist == 'ones':
            torch.nn.init.ones_(m.Lin_1.weight.data)
            torch.nn.init.ones_(m.Lin_2.weight.data)
            torch.nn.init.ones_(m.Lin_3.weight.data)
            torch.nn.init.ones_(m.Lin_4.weight.data)
            torch.nn.init.ones_(m.Lin_5.weight.data)
            
        elif dist == 'normal' and mean is not None and std is not None:
            torch.nn.init.normal_(m.Lin_1.weight.data, mean=mean, std=std)
            torch.nn.init.normal_(m.Lin_2.weight.data, mean=mean, std=std)
            torch.nn.init.normal_(m.Lin_3.weight.data, mean=mean, std=std)
            torch.nn.init.normal_(m.Lin_4.weight.data, mean=mean, std=std)
            torch.nn.init.normal_(m.Lin_5.weight.data, mean=mean, std=std)
            
        if zero_bias: 
            torch.nn.init.zeros_(m.Lin_1.bias.data)
            torch.nn.init.zeros_(m.Lin_2.bias.data)
            torch.nn.init.zeros_(m.Lin_3.bias.data)
            torch.nn.init.zeros_(m.Lin_4.bias.data)
            torch.nn.init.zeros_(m.Lin_5.bias.data)

            
def NN_inputs(model_list, N_list, U_list, mu_list, n_states):
    
     
    all_states = torch.tensor([model_list[k].all_states[i].reshape((1,-1)) 
                                for k in range(len(model_list)) for v in range(int(n_states)) 
                                for j in range(len(U_list)) for i in range( model_list[k].tot_states ) ]).squeeze()

    U_tensor = torch.tensor([U for v in range(int(n_states)) for k in range(len(model_list)) for U in U_list
                              for i in range(model_list[k].tot_states)]).double()
    U_tensor = U_tensor.reshape((U_tensor.shape[0], 1))
    
    mu_tensor = torch.tensor([mu for v in range(int(n_states)) for k in range(len(model_list)) for mu in mu_list 
                              for i in range(model_list[k].tot_states)]).double()
    mu_tensor = mu_tensor.reshape((mu_tensor.shape[0], 1))
    
    N_tensor = torch.tensor([N for v in range(int(n_states)) for (k,N) in enumerate(N_list) for j in range(len(mu_list)) 
                              for i in range(model_list[k].tot_states)]).double()
    N_tensor = N_tensor.reshape((N_tensor.shape[0], 1))
    
    
    state_tensor = torch.tensor([ v for v in range(int(n_states)) for k in range(len(model_list)) for j in range(len(mu_list)) 
                              for i in range(model_list[k].tot_states)]).double()
    
    state_tensor = state_tensor.reshape((state_tensor.shape[0], 1))
    # print(all_states.shape, state_tensor.shape, U_tensor.shape, N_tensor.shape)
    
    UtN_tensor = torch.cat((U_tensor, mu_tensor, N_tensor, state_tensor), 1)
    
    return all_states, UtN_tensor            



def perturb_U(U_list, amp=0.01):
    dU = (torch.rand(U_list.shape)-0.5)*amp
    return U_list + dU
   
    
def train_NN(model_list, N_list, mu_list, U_list, t, V, S, params, fname, 
             use_sampler=False, init=None, loadweights=False, fname_load=None,
             n_excited=0):
    
    M = model_list[0].M
    
    # define default values / reading from the list of parameters
    if 'step_size' in params.keys():
        lr = params['step_size']
    else:
        lr = 1e-3
    if 'max_iteration' in params.keys():
        epochs = params['max_iteration']
    else:
        epochs = 100
    if 'check_point' in params.keys():
        check_point = params['check_point']
    else: 
        check_point = 5
    if 'D_hid' in params.keys():
        D_hid = params['D_hid']
    else:
        D_hid = model_list[0].M
    if 'loss_diff' in params.keys():
        loss_diff = params['loss_diff']
    else:
        loss_diff = 1e-8
    if 'steps' in params.keys(): 
        steps = params['steps']
    else: 
        steps = epochs
    if "loss_check_steps" in params.keys():
        loss_check_steps = params['loss_check_steps']
    else:
        loss_check_steps = steps
    if "grad_cut" in params.keys():
        grad_cut = params["grad_cut"]
    else: 
        grad_cut = 1e-5
    if "grad_clip" in params.keys():
        grad_clip = params["grad_clip"]
    else: 
        grad_clip = None
    if "weight_init" in params.keys() and "init_type" in params.keys(): 
        if_weight_init = True 
        init_type = params["init_type"]
        if "mean" in params.keys(): 
            mean = params["mean"]
        if "std" in params.keys():
            std = params["std"]
        if "zero_bias" in params.keys():
            zero_bias = params["zero_bias"]
        else: 
            zero_bias = False
    else: 
        if_weight_init = False 
    if "proj_steps" in params.keys(): 
        proj_steps = params["proj_steps"]
        intervals = np.arange(0, epochs+proj_steps, proj_steps)
    else: 
        intervals = np.array( [0, epochs] )
    if "gs_flag" in params.keys(): 
        gs_only = params["gs_flag"]
    else: 
        gs_only = True # minimizing the ground state energy by default 
    if "es_flag" in params.keys():
        es_only = params["es_flag"]
    else: 
        es_only = False

    if "regularization" in params.keys(): 
        regularization = params["regularization"]
    else:
        regularization = False
    if "rand_steps" in params.keys():
        rand_steps = params["rand_steps"]
    if "load_states" in params.keys():
        load_states = params["load_states"] # the number of fixed excited states
        assert load_states < n_excited or gs_only 
    else: 
        load_states = 0 
    if "load_states_indv" in params.keys():
        load_states_indv = params["load_states_indv"]
    else: 
        load_states_indv = [0]
    if "load_weights_from_previous_state" in params.keys(): 
        load_weights_from_prev = params["load_weights_from_previous_state"]
    else: 
        load_weights_from_prev = False
    if "use_gpu" in params.keys():
        use_gpu = params["use_gpu"]
    else: 
        use_gpu = False
    if "weight_decay" in params.keys(): 
        weight_decay = params["weight_decay"]
    else: 
        weight_decay = 0
    if "perturb_amp" in params.keys():
        perturb_amp = params["perturb_amp"]
        print(perturb_amp)
    else: 
        perturb_amp = 0.0
    if "data_augment" in params.keys(): 
        data_augment = params["data_augment"]
        if data_augment and "NU_augment" in params.keys(): 
            NU_augment = params["NU_augment"]
        if data_augment and not "NU_augment" in params.keys(): 
            NU_augment = len(U_list)
            print("`NU_augment` not specified. Choosing default value NU_augment = {}".format(NU_augment))
    else:
        data_augment = False
    if "dropout" in params.keys(): 
        dropout = params["dropout"]
    else: 
        dropout = 0 
        
        
    assert load_states_indv[-1] == load_states, print("The last value of `load_states_indv` needs to be equal to the `load_states` (the total states to be fixed)")

        
    fc0 = feedforward(M, D_hid, dropout).double()
  
    betas = [0.999, 0.9999]    
    # betas = [0.9, 0.99]
    # optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas) # SGD, adaptive 
    optimizer = optim.SGD(fc0.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    Loss_history = [] 
    dot_history = []
    
    # load weights from previous runs 
    # if ES:
    #     fname_load = fname_load + str(n_excited)
    if fname_load is not None: 
        if (path.exists(fname_load + "n" + str(n_excited))) and (loadweights==True):
            checkpoint = torch.load(fname_load + "n" + str(n_excited))
            fc0.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            tt = checkpoint['epoch']
            Ltot = checkpoint['loss']
            fc0.train() # or model.eval
            print("Load existing weights...")
    else: 
          if if_weight_init: 
              weight_init(fc0, init_type, mean=mean, std=std, zero_bias=zero_bias)
 
    
    # load weights from the state n-1
    if load_weights_from_prev and (n_excited - load_states == 1):
        if (path.exists(fname + "n" + str(n_excited-1))) :
            checkpoint = torch.load(fname_load + "n" + str(n_excited-1))
            fc0.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            tt = checkpoint['epoch']
            Ltot = checkpoint['loss']
            fc0.train() # or model.eval
            print("Load existing weights...")
        
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps) # Scheduled learning rate

    U_list = torch.tensor(U_list)
    
    if data_augment: 
        U_test = torch.linspace( torch.min(U_list*0.9), torch.max(U_list*1.1), NU_augment)
        load_state_iter = 2
    else: 
        load_state_iter = 1
    
    nn_local = []
    nn_list = []
    loss_local = []
    loss_list = []
    iteration = 0
    wf_gs = []
    stop = False 
    
    if not es_only: 
        gs_opt = True
    
    else: 
        gs_opt = False 
       
        cond = [ path.exists(fname + "n" + str(x)) for x in load_states_indv]
        # print(sum(cond))
        
        # loading excited states wavefunction 
     
                
        if ((load_states and all(cond)) or n_excited==1 or load_states_indv == [0]) and (path.exists(fname + "gs")): 
            
            fc_gs = load_weights(fname + "gs", M, D_hid)
            
            all_states, Ut_tensor = NN_inputs(model_list, N_list, U_list, mu_list, 1)
            u = torch.exp(fc_gs(all_states.double(),Ut_tensor.double())[:,0])
            
            wf_gs_here = torch.zeros((len(u), 1))
            n_states = model_list[0].tot_states
            for i in range(len(U_list)): 
                wf_gs_here[i*n_states:(i+1)*n_states, 0] = normalize(u [i*n_states:(i+1)*n_states])
    
    
            if n_excited > 1 and load_states_indv != [0]:
                # load weights for excited states
                states_list = np.sort(np.append( load_states_indv, 0))
                
                for i,x in enumerate(load_states_indv): 
                    
                    ds = int(states_list[i+1]-states_list[i])
                    fname_here = fname + "n" + str(x)
                    fc_es = load_weights(fname + "n" + str(x), M, D_hid)
                    all_states, Ut_tensor = NN_inputs(model_list, N_list, U_list, mu_list, ds)
                    u_tmp = fc_es(all_states.double(),Ut_tensor.double())
                    u_tmp = u_tmp.reshape( (int(len(u_tmp)/ds), ds) )
                    
                    if i == 0: 
                        
                        for i in range(len(U_list)): 
                            u_tmp[i*n_states:(i+1)*n_states] = Gram_Schmidt( u_tmp[i*n_states:(i+1)*n_states], \
                                                                            wf_gs_here[i*n_states:(i+1)*n_states], use_gpu=use_gpu)
                        
                        u_out = torch.cat( (wf_gs_here, u_tmp), axis=1 )
                  
                        # print(u_tmp.squeeze().float().dot(wf_gs_here.squeeze()))
                    else: 
                    
                        for i in range(len(U_list)):
                            u_tmp[i*n_states:(i+1)*n_states,:] = Gram_Schmidt( u_tmp[i*n_states:(i+1)*n_states,:], \
                                                                              u_out[i*n_states:(i+1)*n_states,:], use_gpu=use_gpu)
                            
                        u_out = torch.cat((u_out, u_tmp),axis=1)
                   
                wf_gs = u_out
            else: 
                wf_gs = wf_gs_here 
    
   
            # wf_tmp = wf_tmp.reshape( ( int(len(wf_tmp)/load_states), int(load_states) ) )
            
            # for x in range(wf_gs.shape[1]):
            #     for y in range(x+1,wf_gs.shape[1]): 
            #         print(wf_gs[:,x].dot(wf_gs[:,y]))
            
            print("Loading weights for ground states previous " + str(load_states+1) + " states")
        
        elif (path.exists(fname + "gs")):
            
            checkpoint = torch.load(fname + "gs")
            fc_gs.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            tt = checkpoint['epoch']
            Ltot = checkpoint['loss']
            fc_gs.train() # or model.eval
            
            all_states, UtN_tensor = NN_inputs(model_list, N_list, U_list, mu_list, 1)
            u = fc_gs(all_states.double(),UtN_tensor.double())
            wf_gs = torch.exp(u[:,0])
            wf_gs = normalize(wf_gs)
            print(wf_gs)
            
            print("Load existing weights for ground states...")
            
        else: 
            gs_opt = True
            print("Ground state weights do not exist. Minimize ground state energy...")
            
    if use_gpu and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        
    tot_states_list = []
    H_list = []
    all_E_list = []
    
    for model in model_list: 
        tot_states_list.append(model.tot_states)
        for (i,U) in enumerate(U_list): 
            _, _, H = model.H_Bose_Hubbard(t, U, V=V, mu=mu_list[i])
            H_list.append(torch.tensor(H, dtype=torch.double))
    
    for tt in range(int(epochs/steps)): 
        optimizer = optim.SGD(fc0.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) # this line is not necessary on my local computer. Not sure why. Without this line learning rate keeps resetting to 0 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        for idx in range(steps):
            if not stop: 
                
                if gs_opt: 
                    n_states = 1 
                else: 
                    n_states = n_excited
                
                # calculate the wavefunction from the nn output
                U_perturb = perturb_U(U_list, amp=perturb_amp)
                # U_perturb = U_list
                
                all_states, UtN_tensor = NN_inputs(model_list, N_list, U_perturb, mu_list, n_states-load_states)
                u = fc0(all_states.double(),UtN_tensor.double())
                
                # real part 
                if gs_opt: 
                    wf_re = torch.exp(u[:,0])
                    # in case wavefunction gets too large
                    wf_norm = torch.sqrt( torch.sum(wf_re.pow(2)))
                    wf_re = wf_re/wf_norm
                else: 
                    wf_re = u[:,0]
                    wf_norm = torch.sqrt( torch.sum(wf_re.pow(2)))
                    wf_re = wf_re/wf_norm
                    
                    # reshape the wavefunction into n x p, where p is the number of excited states
                    wf_re = wf_re.reshape( (int(u.shape[0]/(n_states-load_states)), n_states-load_states) )

                # wf_re[UtN_tensor[:,-1]==0] = torch.exp(u[UtN_tensor[:,-1]==0,0]) # exponential output for the ground state
                # u_im = u[:,1] # imaginary part of the wavefunction exp(1j*u_im)
                
                # normalize the ground state wavefunction
                if gs_opt:
                     wf_norm = torch.zeros(len(wf_re))
                     idx = 0
                
                     for (j, model) in enumerate(model_list):
                         for i in range(len(U_perturb)):
                             wf_here = wf_re[idx+i*int(model.tot_states):idx+(i+1)*int(model.tot_states)]
                             wf_norm[idx+i*int(model.tot_states):idx+(i+1)*int(model.tot_states)] = torch.sqrt(torch.sum(wf_here.pow(2)))
    
                         idx += (i+1)*int(model.tot_states)
                    
                     wf_re = torch.div(wf_re.squeeze(), wf_norm)
                     wf_gs = wf_re.clone()
                            
                
                if not gs_opt: # excited satate
                    seed = np.floor(iteration/rand_steps)
                    
                    if use_gpu and torch.cuda.is_available(): 
                        with torch.cuda.amp.autocast():
                            Ltot, dot_prod, all_E = energy(wf_re.double(), wf_gs.double(), H_list, \
                                                           tot_states_list, mu_list, U_perturb, t, N_list, V, \
                                                           n_states-wf_gs.shape[1]+1, regularization=regularization, \
                                                           seed=seed, use_gpu=use_gpu)
                                                     
                    else:
                        Ltot, dot_prod, all_E = energy(wf_re.double(), wf_gs.double(), H_list, \
                                                       tot_states_list, mu_list, U_perturb, t, N_list, V, \
                                                       n_states-wf_gs.shape[1]+1, \
                                                       regularization=regularization, seed=seed, use_gpu=use_gpu)
                    
                    if use_gpu: 
                        
                        Loss_history.append(Ltot.data.detach().cpu())  
                    else:
                        dot_history.append(dot_prod.detach())
                        all_E_list.append(all_E.detach())
                        Loss_history.append(Ltot.data.detach()) 
                    
                else: # first optimize for the ground state wavefunction
                    if use_gpu: 
                        with torch.cuda.amp.autocast():
                            Ltot = energy_1state(wf_re, H_list, tot_states_list, mu_list, U_perturb, t, N_list, V)
                        Loss_history.append(Ltot.data.detach().cpu())  
                        
                    else:
                        Ltot = energy_1state(wf_re, H_list, tot_states_list, mu_list, U_perturb, t, N_list, V)
                        Loss_history.append(Ltot.data.detach())  
                 
                
                # back-propagating the weights
                if use_gpu: 
                    scaler.scale(Ltot).backward(retain_graph=True)
                    scaler.step(optimizer)
                    scaler.update()
                    
                else: 
                    Ltot.backward(retain_graph=True); 
                    optimizer.step()
     
                
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm(fc0.parameters(), grad_clip)

                if iteration % check_point == 0: 
                    if use_gpu: 
                        grad1=fc0.Lin_1.weight.grad.data.cpu().sum()
                        grad2=fc0.Lin_2.weight.grad.data.cpu().sum()
                        grad3=fc0.Lin_3.weight.grad.data.cpu().sum()
                        try: 
                            grad4=fc0.Lin_4.weight.grad.data.cpu().sum()
                        except: 
                            pass
                        try:
                            grad5=fc0.Lin_5.weight.grad.data.cpu().sum()
                        except:
                            pass
                    else:
                        grad1=fc0.Lin_1.weight.grad.data.sum()
                        grad2=fc0.Lin_2.weight.grad.data.sum()
                        grad3=fc0.Lin_3.weight.grad.data.sum()
                        try: 
                            grad4=fc0.Lin_4.weight.grad.data.sum()
                        except:
                            pass
                        try:
                            grad5=fc0.Lin_5.weight.grad.data.sum()
                        except:
                            pass
                            
                    grad_tot = grad1+grad2+grad3
                    try:
                        grad_tot += grad4
                    except: 
                        pass
                    try: 
                        grad_tot += grad5
                    except:
                        pass
                        
                    print("Iteration {}, current learning rate {}, loss {}, total gradient {}".format(iteration, 
                                                                                                    scheduler.get_lr()[0], 
                                                                                                    Ltot.detach().data,
                                                                                                    grad_tot))
                   
                stop1 = iteration >= epochs-1
                if stop1: 
                    
                    min_idx = np.argmin(loss_list)
                    fc1 = copy.deepcopy(nn_list[min_idx])
                    print(np.min(loss_list))
             
                    print('Maximum iteration reached. Minimum loss not reached.')
              
                    if es_only:       
                        torch.save({
                            'epoch': tt,
                            'model_state_dict': fc1.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': Ltot,
                            }, fname + "n" + str(n_excited))
                    else: # save ground state
                        torch.save({
                            'epoch': tt,
                            'model_state_dict': fc1.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': Ltot,
                            }, fname + "gs")
            
                if (iteration % loss_check_steps == 0) and (iteration > 0): 
                    # find the minimum loss of every 500 steps (or other custom number of steps)
                    min_idx = np.argmin(loss_local)
                    nn_list.append(nn_local[min_idx])
                    loss_list.append(loss_local[min_idx])

                    # check if the stopping criterion is satisfied
                    if int(iteration / loss_check_steps) > 2:
                        stop2 = (np.std(loss_local) < loss_diff) 
                    
                        stop3 = (np.abs(np.mean(loss_local)-loss_local[-1]) < loss_diff)  
       
                        stop4 = (np.abs(grad_tot) < grad_cut)  
                        
                        stop5 = (np.abs(np.mean(loss_local)) < loss_diff)  
                        
                        if (stop2 and stop3) or stop4 or stop5: 
                            min_idx = np.argmin(loss_list)
                            fc1 = copy.deepcopy(nn_list[min_idx])
                            print(np.min(loss_list))
                            
                            if (stop2 and stop3): 
                                print("Minimum loss reached")
                                
                                stop = True # stop 
                                
                            elif stop4: 
                                print("Minimum gradient reached")
                                stop = True
                                    
                                    
                            if es_only:       
                                torch.save({
                                    'epoch': tt,
                                    'model_state_dict': fc1.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': Ltot,
                                    }, fname + "n" + str(n_excited))
                            else: # save ground state
                                torch.save({
                                    'epoch': tt,
                                    'model_state_dict': fc1.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': Ltot,
                                    }, fname + "gs")
                    
                                # print(wf_gs)
                                print("Finish minimizing the ground state energy.")
                              
  
                        
                        
                    nn_local = []
                    loss_local = []
                    
                    if use_gpu and not gs_opt: 
                        dot_history.append(dot_prod.detach().cpu())
                        all_E_list.append(all_E.detach().cpu())
                    
                else: 
                    loss_local.append(Loss_history[-1])
                    nn_local.append(copy.deepcopy(fc0))
                
                scheduler.step() 
                optimizer.zero_grad() 
                    
                iteration += 1
            
        # print('Reset scheduler')
        
        
    
    return fc1, Loss_history, dot_history, all_E_list



def load_weights(fname, M, D_hid, dropout=0): 

    fc = feedforward(M, D_hid, dropout).double()
    
    if (path.exists(fname)):
        checkpoint = torch.load(fname)
        fc.load_state_dict(checkpoint['model_state_dict'])
        fc.train()
        print("Loading existing weights...")
        
    return fc 



def wf_e_calc(model, N, U_list, mu_list, t, V, n_excited, load_states_indv, fc_gs, fc1=None, fc2=None, use_gpu=False): 
    '''
    Calculate the wavefunction and energy from the NN output 
    fc_gs: network for ground state
    fc1: network for excited states (optimized)
    fc2: a list of networks for excited states (fixed)
    '''
    
    load_states = np.max(load_states_indv)
    
    E = np.zeros( (len(U_list), int(n_excited+1)) )
    
    all_states, Ut_tensor = NN_inputs([model], [N], U_list, mu_list, 1)
    
    wf_gs = torch.exp(fc_gs(all_states.double(),Ut_tensor.double()))
    
    wf_norm = torch.zeros(len(wf_gs))
    idx = 0
   
    # normalization for each U
    for i in range(len(U_list)):
        wf_here = wf_gs[i*int(model.tot_states):(i+1)*int(model.tot_states)]
        wf_norm[i*int(model.tot_states):(i+1)*int(model.tot_states)] = torch.sqrt(torch.sum(wf_here.pow(2)))
   
    wf_gs = torch.div(wf_gs.squeeze(), wf_norm)
    wf_gs = wf_gs.reshape( (len(wf_gs),1) )
    H_list = [] 
    n_state_list = []
 
    for (i, U) in enumerate(U_list):
        _, _, H = model.H_Bose_Hubbard(t, U, V=V, mu=mu_list[i])
        H_list.append(torch.tensor(H, dtype=torch.double))
        n_state_list.append(model.tot_states)
    
    # E[:,0] = np.array([energy_1state(wf_gs[i*int(model.tot_states):(i+1)*int(model.tot_states)], [H_list[i]], [model.tot_states], \
                                     # mu_list, [U], float(t), [N], V).detach().numpy() for (i,U) in enumerate(U_list)]) 
    
    # E_mean = energy_1state(wf_gs, H_list, n_state_list, mu_list, U_list, float(t), [N], 0.0)

    E[:,0] = np.array([energy_1state(wf_gs[i*model.tot_states:(i+1)*model.tot_states], [H_list[i]], [model.tot_states], [mu_list[0]], [U], float(t), [N], V).cpu().detach().numpy()
                       for (i, U) in enumerate(U_list)] )
    
    if n_excited == 0: 
        return E, wf_gs

    else: 
        all_states, Ut_tensor = NN_inputs([model], [N], U_list, mu_list, n_excited-np.max(load_states_indv))
        u_out = fc1(all_states.double(),Ut_tensor.double())
        
        
        if fc2 is not None and load_states_indv != [0]: 
            
            states_list = np.sort(np.append( load_states_indv, 0))
            
            for i,x in enumerate(load_states_indv): 
                
                fc_es = fc2[i]
                all_states, Ut_tensor = NN_inputs([model], [N], U_list, mu_list, states_list[i+1]-states_list[i])
                u_tmp = fc_es(all_states.double(),Ut_tensor.double())
                u_tmp = u_tmp.reshape( (int(len(u_tmp)/(states_list[i+1]-states_list[i])),  states_list[i+1]-states_list[i]) )
                
                u_out2_tmp = torch.zeros_like(u_tmp)
                # print(u_tmp.shape)
        
                if i == 0: 
                
                    for i in range(len(U_list)): 
                        
                        u_out2_tmp[i*model.tot_states:(i+1)*model.tot_states, :] = \
                            Gram_Schmidt(u_tmp[i*model.tot_states:(i+1)*model.tot_states,:], \
                                         wf_gs[i*model.tot_states:(i+1)*model.tot_states,:], use_gpu=use_gpu)
                    
                    u_out2 = torch.cat((wf_gs,u_out2_tmp), axis=1)
                    
                else: 
                    
                    # u_out2 = torch.zeros_like(u_tmp)
                    for i in range(len(U_list)): 
                        
                        u_out2_tmp[i*model.tot_states:(i+1)*model.tot_states, :] = \
                            Gram_Schmidt(u_tmp[i*model.tot_states:(i+1)*model.tot_states,:],\
                                         u_out2[i*model.tot_states:(i+1)*model.tot_states,:], use_gpu=use_gpu)
                    
                    u_out2 = torch.cat((u_out2, u_out2_tmp),axis=1)
            
            
            wf_gs = u_out2
        
        # check orthogonality

        # print(u_out2.shape)
        # for x in range(u_out2.shape[1]):
        #     for y in range(x+1,u_out2.shape[1]):
        #         if x != y: 
        #             print(u_out2[:,x].dot(u_out2[:,y]))

        
        # new part 
        wf_tmp = u_out[:,0]
        wf_tmp = wf_tmp.reshape( (int(u_out.shape[0]/(n_excited-load_states) ), int(n_excited-load_states) ) )
        
        wf = torch.zeros( (wf_tmp.shape[0], int(n_excited+1)) )
 
        idx = 0 
        
        # orthogonalize the wavefunction
        for i in range(len(U_list)):
            
            wf_gs_here = wf_gs[idx+i*int(model.tot_states):idx+(i+1)*int(model.tot_states), :]
            # wf_here[:,0] = wf_single_proj(wf_gs[idx+i*int(model.tot_states):idx+(i+1)*int(model.tot_states)], wf_here[:,0])
            tmp = wf_tmp[idx+i*int(model.tot_states):idx+(i+1)*int(model.tot_states), :]
            
            wf_here = Gram_Schmidt(tmp, wf_gs_here, use_gpu=use_gpu)

            
            for s in range(wf_gs_here.shape[1]): 
                tmp = energy_1state(wf_gs_here[:,s], [H_list[i]], [model.tot_states], [mu_list[i]], [U_list[i]], t, [N], V)
                E[i, s] = tmp
                
            e_idx = np.argsort( E[i, 0:wf_gs_here.shape[1]]) 
            wf_gs_tmp = wf_gs_here[:,e_idx]
            E[i,0:wf_gs_here.shape[1]] = np.sort( E[i, 0:wf_gs_here.shape[1]] )
        
    
            for s in range(wf_here.shape[1]):
                tmp = energy_1state(wf_here[:,s], [H_list[i]], [model.tot_states], [mu_list[i]], [U_list[i]], t, [N], V)
                E[i, wf_gs_here.shape[1]+s] = tmp
                
            wf_all = torch.cat((wf_gs_tmp, wf_here), axis =1)
            # print(wf_all.shape)
            
            # e_idx = np.argsort( E[i, :]) 
            # E[i, :] = E[i, e_idx]
        
            
            wf[idx+i*int(model.tot_states):idx+(i+1)*int(model.tot_states),:] = wf_all
            # idx += (i+1)*int(model.tot_states)
            
        return E, wf



