import numpy as np

def DFT(x): #https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
	N = len(x)
	n = np.arange(N)
	k = n.reshape((N,1))
	e = np.exp(-2j * np.pi * k *n /N)
	X = np.dot(e,x)
	return X

def DFT_inverse(X):
	N = len(X)
	n = np.arange(N)
	k = n.reshape((N,1))
	e = np.exp(2j*np.pi * k *n /N)
	x = np.dot(e,X) /N
	return x

def padd_pow2(x):
	N = len(x)
	l = np.log2(N)
	pa =2**np.ceil(l) -N
	pad = np.zeros(int(pa))
	x_new = np.concatenate([x,pad],axis=0)
	return x_new

def FFT_real(x):
	N = len(x) 
	n = np.arange(N)
	if N== 1:
		return x 
	else: 
		X_even = FFT(x[::2])
		X_odd = FFT(x[1::2])
		factor = np.exp(-2j*np.pi*n /N)
		X = np.concatenate([X_even + factor[:int(N/2)]*X_odd, X_even+factor[int(N/2):]*X_odd])
		return X

### This function takes any old array and prepares it for the fourier transform, should be updated to transform 2d => 1d => 2d
def FFT(x):
	return FFT_real(padd_pow2(x))


x= np.array([-2,4,2])
X = FFT(x)
print(DFT_inverse(X).real)