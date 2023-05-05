import numpy as np
from joblib import Parallel, delayed

###Filter Banks

##Harr

def harr_a_filters(N):
	h, l = np.zeros(N), np.zeros(N)
	h[0], h[-1], l[0], l[-1]= .5,-.5, .5,.5
	return l,h

def harr_s_filters(N):
	h, l = np.zeros(N), np.zeros(N)
	h[0], h[-1], l[0], l[-1]= 1,-1, 1,1
	return l,h

def daubecies_a_filters(N):
	h,l = np.zeros(N), np.zeros(N)
	h[0],h[-1],h[-2],h[-3] = (1 + np.sqrt(3))/8, (3+np.sqrt(3))/8, (3-np.sqrt(3))/8, (1-np.sqrt(3))/8
	l[-1],l[0],l[1],l[2] = -1*(1 + np.sqrt(3))/8, (3+np.sqrt(3))/8, -1*(3-np.sqrt(3))/8, (1-np.sqrt(3))/8
	return l,h

def padded(f):
	N = f.size
	return np.concatenate((f,np.zeros(N)))

def conv(f,g,n):
	f,g = padded(f),padded(g)
	N = f.size
	conv = 0
	for m in range(N):
		conv += f[m]*g[n-m]
	return conv

def downsample(f,n):
	N = int(f.size/n)
	g = np.zeros(N)
	for i in range(N):
		g[i]=f[n*i]
	return g

def filter_bank_step(f,filter):
	g= np.array(Parallel(n_jobs=-1)([delayed(conv)(f,filter,i) for i in range(f.size)]))
	return downsample(g,2)


def DWT(f,filter,lev):
	M = f.shape[0]
	N = f.shape[1]
	l_c,h_c = filter(M)
	coeffs_lc = np.transpose(np.array(Parallel(n_jobs=-1)([delayed(filter_bank_step)(f[:,i],l_c) for i in range(N)])))
	coeffs_hc = np.transpose(np.array(Parallel(n_jobs=-1)([delayed(filter_bank_step)(f[:,i],h_c) for i in range(N)])))
	#for i in range(N):
		#coeffs_lc[:,i] = filter_bank_step(f[:,i],l_c)
		#coeffs_hc[:,i] =  filter_bank_step(f[:,i],h_c)
	l_r,h_r = filter(N)
	M,N = int(M/2),int(N/2)
	coeffs_lchr, coeffs_hclr, coeffs_hchr = np.zeros((M,N)),np.zeros((M,N)),np.zeros((M,N))
	coeffs_lclr = np.array(Parallel(n_jobs=-1)([delayed(filter_bank_step)(coeffs_lc[i,:],l_r) for i in range(M)]))
	coeffs_lchr = np.array(Parallel(n_jobs=-1)([delayed(filter_bank_step)(coeffs_lc[i,:],h_r) for i in range(M)]))
	coeffs_hclr = np.array(Parallel(n_jobs=-1)([delayed(filter_bank_step)(coeffs_hc[i,:],l_r) for i in range(M)]))
	coeffs_hchr = np.array(Parallel(n_jobs=-1)([delayed(filter_bank_step)(coeffs_hc[i,:],h_r) for i in range(M)]))
	#for i in range(M):
		#coeffs_lclr[i,:] = filter_bank_step(coeffs_lc[i,:],l_r)
	#	coeffs_lchr[i,:] =  filter_bank_step(coeffs_lc[i,:],h_r)
	#	coeffs_hclr[i,:] = filter_bank_step(coeffs_hc[i,:],l_r)
	#	coeffs_hchr[i,:] =  filter_bank_step(coeffs_hc[i,:],h_r)
	if(lev == 1):
		return coeffs_lclr, coeffs_lchr, coeffs_hclr, coeffs_hchr 
	else:
		return DWT(coeffs_lclr, filter,lev-1), coeffs_lchr, coeffs_hclr, coeffs_hchr 