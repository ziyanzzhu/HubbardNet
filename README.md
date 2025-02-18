# HubbardNet: efficient predictions of the Bose-Hubbard Model Spectrum with Deep Neural Networks

All scripts have been tested on Google Colab (have options for both CPU/GPU). Can also run locally. 


## Citation
For reference to the `HubbardNet` model, please see and cite the following manuscript: 

"HubbardNet: Efficient Predictions of the Bose-Hubbard Model Spectrum with Deep Neural Networks." 

Ziyan Zhu, Marios Mattheakis, Weiwei Pan, Efthimios Kaxiras

Phys. Rev. Research 5, 043084 (2023) | ArXiv: 2212.13678

## Contact 
Ziyan (Zoe) Zhu: ziyanzhu [at] stanford.edu

Please contact me with any issues and/or requests.



## List of files: 
- `matrix_elements.py`: functions to construct the Bose-Hubbard model 
- `HubbardNet_gpu.py`: functions to construct the DNN and train the network
### Examples:
- `iterative_opt_gpu_for_gs.ipynb`: ground state optimization 
- `iterative_opt_gpu_for_gs_mult_N.ipynb`: ground state optimization with multiple N's in the training set
- `iterative_opt_gpu_spectrum.ipynb`: iteratively obtain the full spectrum (all excited states) using multiple U's in the training set
