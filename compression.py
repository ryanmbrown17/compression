import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

##Base 
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


#### Color Spaces
###BT.601
def BT_601(RGB):
	R = RGB[0]
	G = RGB[1]
	B = RGB[2]
	Y= 0.299*R + 0.587*G + 0.114*B
	CR = (R - Y)/1.402
	CB = (B- Y)/1.772
	return (Y,CR,CB)

def Pix_transform(pix,T): 
	return np.matmul(pix,T)

def Transform(Kr, Kb):
	Kg = 1 - Kr - Kb
	r1 = [Kr, Kg, Kb]
	r2 = [-Kr / (2-2*Kb), -Kg/(2-2*Kb),1/2]
	r3 = [1/2, -Kg/(2-2*Kr), -Kb/(2-2*Kr)]
	return np.array([r1,r2,r3])

def Transform_BT(K_ry, K_by):
	K_gy = 1 - K_ry - K_by
	K_ru = -K_ry
	K_gu = -K_gy
	K_bu = 1- K_by
	K_rv = 1- K_ry
	K_gv = -K_gy
	K_bv = - K_by
	return np.array([[K_ry, K_gy, K_by],[K_ru,K_gu,K_bu],[K_rv,K_gv,K_bv]])

T = Transform(0.2989,0.1140)
T_inv = np.linalg.inv(T)
full_T=np.matmul(T_inv,T)
#print(T)
#print(T_inv)
#print(full_T) 
#im = Image.open('happy-test-screen.jpg','r')
im = Image.open('mario.png','r')


pixels = np.array(im)

##Remove gamma
__rg = lambda x: x[:,:,0:3]
_rg = lambda x:__rg(x)
rg = lambda x:_rg(x)

__t = lambda x: Pix_transform(x,T)
_t = lambda x: __t(x)
t = lambda x: _t(x)

pixels = rg(pixels)



transform = t(pixels)
h = transform[:,:,2]
s = transform[:,:,1]
v = transform[:,:,0]

output = [im, h, s, v]
title = ['Image','Hue','Saturation','Value']
        
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.axis('off')
    plt.title(title[i])
    if i == 0:
        plt.imshow(output[i])
    else:
        plt.imshow(output[i], cmap = 'gray')     
plt.show()





#plt.imshow(gray,cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#plt.show()
