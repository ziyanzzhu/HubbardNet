#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:53:54 2021

@author: zoe

Calculate the matrix element of 1D Bose-Hubbard model

"""

import numpy as np
from math import *
from scipy.sparse import csr_matrix 
from scipy.special import comb


class Bose_Hubbard(): 
    '''
    Construct Bose Hubbard Model 
    N : Number of particles
    M : Number of sites / orbitals, optional for triangular lattices 
        since M is determined by w and h
    O : Maximum number of particles per site (= N for bosons, = 2 for fermions)
    w : Number of sites per row for a square lattice 
        if triangular lattice, w = n_L1, the number of sites
    h : Number of sites per column for a square lattice 
        if triangular lattice, the number of layers (NEEDS TO BE AN ODD NUMBER)
    lattice : Type of lattices:
              "square": square lattice, including 1D chain 
              "triangular" : triangular lattice, if triangular lattice, periodic boundary condition only
    pbc : Periodic boundary condition
    ''' 
    
    def __init__(self, N, O, M=5, w=5, h=1, lattice="square", pbc=True):
        self.M = M 
        self.N = N
        self.O = O
        self.w = w 
        self.h = h # default 1, h > 1 2D square lattice
        self.pbc = pbc
        self.lattice = lattice
        
        
        # define triangular lattice geometry 
        if lattice[0] == "t":
            # define a cluster
            n_L1 = w # number of site in the first layer 
            k = (h-1)/2
            self.pbc = False
            
            # number of sites per layer
            n_L = np.zeros( h )
            n_L[0] = n_L1
            
            for m_idx in range(h-1):
                if m_idx < k: 
                    n_L[m_idx+1] = n_L[m_idx] + 1
                else: 
                    n_L[m_idx+1] = n_L[m_idx] - 1
            
            M = int(np.sum(n_L)) # total number of sites
     
            # label site by the layer it's in 
            m_count = np.cumsum(n_L)
            layer_number = np.zeros( M, dtype=int)
            
            for m_idx in range(M):
                flag = 1
                for i,m2 in enumerate(m_count): 
                    if flag:
                        if i==0:
                            m1 = 0 
                        else:
                            m1 = m_count[i-1]
                            
                        if (m_idx+1) <= m2 and (m_idx+1)>m1:
                            layer_number[m_idx] = i+1
                            flag = 0
            
            # find the NN of every atoms 
            nn_list = np.zeros((M, 6), dtype=int)
            
            for m_idx in range(M):
                m = m_idx + 1
                layer_here = layer_number[m_idx]
                nL_here = n_L[layer_here-1]
                
                if layer_here-2>=0:
                    nL_previous = n_L[layer_here-2]
                    nn_m = np.array([ m-1, m+1, m+nL_here, m+nL_here+1, m-nL_previous, m-nL_previous-1 ], dtype=int)
                else:
                    nn_m = np.array([ m-1, m+1, m+nL_here, m+nL_here+1 ], dtype=int)
                    
               # site number of 6 NN 
                nn_m = nn_m[nn_m>0]
                nn_m = np.unique(nn_m[nn_m<=M])
                
                idx=0
                for nn in nn_m:
                    if nn == m-1 or nn == m+1:
                        if layer_number[nn-1] == layer_number[m_idx]: 
                            nn_list[m_idx,idx] = nn
                            idx+=1
                    elif nn == m+nL_here or nn == m+nL_here+1: 
                        if layer_number[nn-1] == layer_number[m_idx]+1:
                            nn_list[m_idx,idx] = nn
                            idx+=1
                    elif nn == m-nL_previous or nn == m-nL_previous-1 and layer_here-2 >= 0:
                        if layer_number[nn-1] == layer_number[m_idx]-1:
                            nn_list[m_idx,idx] = nn
                            idx+=1
            
            self.n_L = n_L
            self.M = M
            self.layer_number = layer_number
            self.nn_list = nn_list
            
            
        self.tot_states = int(comb(M+N-1, M-1)) # total number of combinations for bosons
        self.big_N = self.N_table(M, N, O)
        self.all_states = []
        self.all_states = self.find_all_states()
         
        
        if M != w*h and lattice[0] == "s":
            print("The number of sites is not consistent! wh != M")
            
        if np.floor(h/2) == h/2 and lattice[1] == "t":
            print("")
            
        
        
    
    def N_table(self, M, N, O): 
        big_N = np.zeros((M+1, N+1), dtype=int)
        big_N[0, 0] = 1
        for n in range(N+1): 
            for m in range(M+1): 
                if m > 0:
                    for o in range(O+1):
                        big_N[m, n] += big_N[m-1, N-o] 
        self.big_N = np.array(big_N)
        return self.big_N
    
    
    
    def idx_to_fock(self, M, N, O, big_N, n_beta):
        '''
        
        Convert integer label of a state to Fock state. 
        
        Parameters
        ----------
        N : Number of particles
        M : Number of sites / orbitals
        O : Maximum number of particles per site (= N for bosons, = 2 for fermions)
        big_N: Table
        n_beta : Integer label of the state, n_beta = [1, n_max]
    
        Returns
        -------
        fock: Particle occupation in Fock basis
    
        '''
        
        fock = np.zeros(M, dtype=int)
        m_list = np.zeros(N, dtype=int)
        n_beta_tmp = n_beta-1
        for i in range(N):
            n = N-i
            big_N_here = big_N[:,n] 
            rule = big_N_here <= n_beta_tmp
            
            if np.sum(rule) < 1: 
                mj = 0
            else: 
                N_here_idx = np.max(big_N_here[rule])
                mj = np.argmin(np.abs(big_N_here - N_here_idx))
                # print(big_N_here, N_here_idx, mj)
            
            m_list[i] = M - mj
            n_beta_tmp -= big_N_here[mj]
        # print(m_list)
        # convert stie number to fock state
        fock = [np.sum(m_list==m+1) for m in range(M) ]
        fock = np.array(fock)
                
        
        return fock
        
    def fock_to_idx(self, M, N, O, big_N, fock):
        '''
        Convert Fock state to integer label 
    
        Parameters
        ----------
        N : Number of particles
        M : Number of sites / orbitals
        O : Maximum number of particles per site (= N for bosons, = 2 for fermions)
        big_N: Table
        fock : Particle occupation in Fock basis
    
        Returns
        -------
        n_beta : Integer label of the state, n_beta = [1, n_max]
    
        '''
        
        
        mj = []
        for i in range(M): 
            tmp_list = int(fock[i])*[i+1]
            for ele in tmp_list:
                mj.append(ele)
            
        mj = np.array(mj)
        mj = np.sort(mj)[::-1]
        
        big_N_list = [big_N[M-mj[i],i+1] for i in range(N)] # note the index (i+1)
        n_beta = np.sum(big_N_list) + 1
        
        return n_beta
    
    
    def find_all_states(self): 
        for n_beta in range(1, self.tot_states+1): 
            self.all_states.append( self.idx_to_fock(self.M, self.N, self.O, self.big_N, n_beta) )
        return self.all_states 
    
    
    def H_Bose_Hubbard(self, t, U, mu=0, V=0):
        '''
        
        Parameters
        ----------
        t : Hopping parameter
        U : Coulomb parameter
        V : Site-dependent confinement potential
    
        Returns
        -------
        H : Hamiltonian matrix elements in real space
    
        '''
    
        M = self.M 
        N = self.N
        O = self.O
        w = self.w
        h = self.h
        big_N = self.big_N
        n_max = self.tot_states
    
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        mat_elem = []
        for n_beta in range(1,n_max+1):
            fock = self.all_states[n_beta-1]
            # print(n_beta, fock)
            
            vhop = 0 
            hhop = 0
            # find t-terms
            for (j,nj) in enumerate(fock):
            
                cond2 = (np.floor( j/w ) and self.h>1)
                
                # horizontal 
                if j%w: 
                    hop_to_idx = j-1
                    hhop = 1           
                elif not(j%w) and self.pbc: 
                    hop_to_idx = j+w-1 
                    if hop_to_idx > 0:
                        hhop = 1
                
                if hhop: 
                    # hopping between j and (j-1)
                    if (nj > 0): 
                        # a_{j-1}^+ a_j 
                        fock2 = np.copy(fock)
                        fock2[j] -= 1 
                        fock2[hop_to_idx] += 1 
                        n_beta2 = self.fock_to_idx(M, N, O, big_N, fock2)
                        # print(fock)
                        # print(fock2)
                        # print(j)
                        # print("-------------")
                        
                        idx1 = np.argwhere(rows == n_beta-1)
                        idx2 = np.argwhere(cols == n_beta2-1)
                        common_element = [x for x in idx1 if x in idx2] 
                        if len(common_element)==0 : 
    
                            rows = np.append(rows,n_beta-1)
                            cols = np.append(cols,n_beta2-1)
                            mat_elem.append(-np.sqrt(nj)*np.sqrt(fock[hop_to_idx]+1)*t)
                        
                            # print(j, hop_to_idx, n_beta-1, n_beta2-1, -np.sqrt(nj)*np.sqrt(fock[j-1]+1)*t)
                            # print("-----------")
                    
                    if (fock[hop_to_idx] > 0):
                        # a_j^+ a_{j-1}             
                        fock3 = np.copy(fock)
                        fock3[j] += 1 
                        fock3[hop_to_idx] -= 1 
                        n_beta3 = self.fock_to_idx(M, N, O, big_N, fock3)
                        
                        idx1 = np.argwhere(np.abs(rows-n_beta+1)<1e-3)
                        idx2 = np.argwhere(np.abs(cols-n_beta3+1)<1e-3)

                        common_element = [x for x in idx1 if x in idx2] 

                        if len(common_element)==0 : 
                            rows = np.append(rows,n_beta-1)
                            cols = np.append(cols,n_beta3-1)
                            mat_elem.append(-np.sqrt(nj+1)*np.sqrt(fock[hop_to_idx])*t)
                        
                            # print(j, hop_to_idx, n_beta-1, n_beta3-1, -np.sqrt(nj+1)*np.sqrt(fock[j-1])*t)
                            # print("-----------")
                
                # vertical
                if cond2:
                    # no periodic boundary condition 
                    # hopping between j and (j-w)
                    hop_to_idx = j-w
                    vhop = 1
                        
                elif (self.h > 1) and (not np.floor( j/w )) and self.pbc: 
                    hop_to_idx = (h-1)*w + j
                    vhop = 1
                    
                if vhop: 
                    
                    if (nj > 0): 
                        # a_{j-w}^+ a_j 
                        fock2 = np.copy(fock)
                        fock2[j] -= 1
                        fock2[hop_to_idx] += 1
                        n_beta2 = self.fock_to_idx(M, N, O, big_N, fock2)
                        
                        idx1 = np.argwhere(rows == n_beta-1)
                        idx2 = np.argwhere(cols == n_beta2-1)
                        common_element = [x for x in idx1 if x in idx2] 
                        
                        if len(common_element)==0 : 
    
                            rows = np.append(rows,n_beta-1)
                            cols = np.append(cols,n_beta2-1)
                            mat_elem.append(-np.sqrt(nj)*np.sqrt(fock[hop_to_idx]+1)*t)
                            
                        # print(n_beta-1, n_beta2-1, -np.sqrt(nj)*np.sqrt(fock[hop_to_idx]+1)*t)
                        # print("-----------")
                    
                    if (fock[hop_to_idx] > 0) and (self.h > 2): 
                        # a_j^+ a_{j-w}
                        fock3 = np.copy(fock)
                        fock3[j] += 1 
                        fock3[hop_to_idx] -= 1 
                        n_beta3 = self.fock_to_idx(M, N, O, big_N, fock3)
                        
                        idx1 = np.argwhere(rows == n_beta-1)
                        idx2 = np.argwhere(cols == n_beta3-1)
                        common_element = [x for x in idx1 if x in idx2] 
                        if len(common_element)==0 : 
                            rows = np.append(rows,n_beta-1)
                            cols = np.append(cols,n_beta3-1)
                            mat_elem.append(-np.sqrt(nj+1)*np.sqrt(fock[hop_to_idx])*t)
                    
                        # print(n_beta-1, n_beta3-1, -np.sqrt(nj+1)*np.sqrt(fock[hop_to_idx]))
                        # print("-----------")
                
                vhop = 0 
                hhop = 0
            # find U-terms
            rows = np.append(rows, n_beta-1)
            cols = np.append(cols, n_beta-1)
            UVmu_term = np.sum(fock**2)*U/2 - N*U/2 - N*mu
            tmp_arr = [V*(j - (M-1)/2 )**2 * fock[j] for j in range(M)]
            UVmu_term += np.sum(tmp_arr) 
            mat_elem.append(UVmu_term)
            
        # rows = int()
        # cols = np.array(cols)
        
        H = np.zeros( (self.tot_states, self.tot_states), dtype = complex)
        for i in range(len(rows)):
            H[rows[i], cols[i]] = mat_elem[i]
        # H = csr_matrix((mat_elem, (rows, cols)), dtype = float) # somehow this doesn't work for Google Colab for python 3.7
    
        return rows, cols, H 
    
    
    def H_Bose_Hubbard_tri(self, t, U, mu=0, V=0):
        all_states = self.all_states
        M = self.M 
        N = self.N 
        O = self.O 
        big_N = self.big_N 
        nn_list = self.nn_list 
        
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        mat_elem = []
        
        for i, fock in enumerate(all_states): 
            n_beta = self.fock_to_idx(M, N, O, big_N, fock)
            
            # hopping terms
            for j, ni in enumerate(fock): 
                # find the list of nearest neighbors 
                nn_here = nn_list[j]
                nn_here = nn_here[nn_here > ni] # looking at only site numbers greater than itself 
                
                for k in nn_here:
                    nn_idx = k-1
                    
                    if fock[j] > 0:
                        # a_{nn_idx}^+ a_j, hopping from this site to NN site
                        
                        fock2 = np.copy(fock)
                        fock2[nn_idx] += 1
                        fock2[j] -= 1
                    
                        fock3 = np.copy(fock)
                        n_beta2 = self.fock_to_idx(M, N, O, big_N, fock2)
                        # print(fock)
                        # print(fock2)
                        # print(j)
                        # print("-------------")
                        
                        idx1 = np.argwhere(rows == n_beta-1)
                        idx2 = np.argwhere(cols == n_beta2-1)
                        common_element = [x for x in idx1 if x in idx2] 
                        if len(common_element)==0 : 
            
                            rows = np.append(rows,n_beta-1)
                            cols = np.append(cols,n_beta2-1)
                            mat_elem.append(-np.sqrt(ni)*np.sqrt(fock[nn_idx]+1)*t)
                         
                    if fock[nn_idx] > 0:
                        # a_{j}^+ a_{nn_idx}, hopping from an NN site to this iste
                    
                        fock3 = np.copy(fock)
                        fock3[j] += 1 
                        fock3[nn_idx] -= 1 
                        n_beta3 = self.fock_to_idx(M, N, O, big_N, fock3)
                        
                        idx1 = np.argwhere(np.abs(rows-n_beta+1)<1e-3)
                        idx2 = np.argwhere(np.abs(cols-n_beta3+1)<1e-3)
            
                        common_element = [x for x in idx1 if x in idx2] 
            
                        if len(common_element)==0 : 
                            rows = np.append(rows,n_beta-1)
                            cols = np.append(cols,n_beta3-1)
                            mat_elem.append(-np.sqrt(ni+1)*np.sqrt(fock[nn_idx])*t)
                            
                # -U n_i n_i terms
                rows = np.append(rows, n_beta-1)
                cols = np.append(cols, n_beta-1)
                UVmu_term = np.sum(fock**2)*U/2 - N*U/2 - N*mu
                # tmp_arr = [V*(j - (M-1)/2 )**2 * fock[j] for j in range(M)]
                # UVmu_term += np.sum(tmp_arr) 
                mat_elem.append(UVmu_term)
        
        H = csr_matrix((mat_elem, (rows, cols)), dtype = float)  
        return rows, cols, H
                
    # construct Fermi Hubbard model
    def H_Fermi_Hubbard():
        return None
    
