import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from joblib import Parallel, delayed

def show_decomp_lev_1(s1,w11,w12,w13):
	output = [s1,w11,w12,w12]
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.axis('off')
		plt.imshow(output[i], cmap = 'gray')    
	plt.show()

def Number_Transform(num,X,Δ):
	for i in range(X.size):
		if(num < X[i] + Δ/2):
			return i
	#print("BAD PLACE")
	return

def Scalar_Quantization(im,Δ,x0,N):
	print(Δ)
	x = np.zeros(N)
	x[0] = x0
	for i in range(1,N):
		x[i] = x[i-1] + Δ
	im2 = np.zeros(im.size)
	s =0
	for i in range(im.size):
		im2[i] =Number_Transform(im[i],x,Δ)
		s += im2[i] 
	avg = s/im.size
	return im2

def _Quantize(w1,N):
	lil = w1.min()
	big = w1.max()
	x0 = lil+Δ/2
	Δ = (big-lil)/N
	W1 =  np.array(Parallel(n_jobs=-1)([delayed(Scalar_Quantization)(w1[i,:],x0,Δ,N) for i in range(w1.shape[0])]))

def Quantize(w1,w2,w3,N):
	w1 = _Quantize(w1,N)
	w2  = _Quantize(w2,N)
	w3  = _Quantize(w3,N)
	return w1,w2,w3
