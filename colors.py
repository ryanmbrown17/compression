import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

##Utility
def Pix_transform(pix,T): 
	return np.matmul(pix,T)

def Color_Matrix_YUV(Kr, Kb):
	Kg = 1 - Kr - Kb
	r1 = [Kr, Kg, Kb]
	r2 = [-Kr / (2-2*Kb), -Kg/(2-2*Kb),1/2]
	r3 = [1/2, -Kg/(2-2*Kr), -Kb/(2-2*Kr)]
	return np.array([r1,r2,r3])

def Color_Matrix_BT(K_ry, K_by):
	K_gy = 1 - K_ry - K_by
	K_ru = -K_ry
	K_gu = -K_gy
	K_bu = 1- K_by
	K_rv = 1- K_ry
	K_gv = -K_gy
	K_bv = - K_by
	return np.array([[K_ry, K_gy, K_by],[K_ru,K_gu,K_bu],[K_rv,K_gv,K_bv]])

def Color_Matrix_RCT():
	r1 = [.25,.5,.25]
	r2 = [0,-1,1]
	r3 = [1,-1,0]
	return np.array([r1,r2,r3])
##Removing Gamma Component
__rg = lambda x: x[:,:,0:3]
_rg = lambda x:__rg(x)
rg = lambda x:_rg(x)


##Transform Each set of pixels
__t = lambda x,T: Pix_transform(x,T)
_t = lambda x,T: __t(x,T)
t = lambda x,T: _t(x,T)

###Actual Functions

##Takes an Image and returns Intensity(values), Cb, and Cr
## Image value is given by pixels, Kr and Kb represent the values of the specific color transform, Type if 0 means The color matrix where as 1 is the BT
def transform(Kr, Kb, Type,Im):
	if(Type == 0):
		T= Color_Matrix_YUV(Kr,Kb)
	else:
		T= Color_Matrix_BT(Kr,Kb)
	pixels = rg(np.array(Im))
	transform = t(pixels,T)
	Pr = np.array(transform[:,:,2]) ## or Hue
	Pb = np.array(transform[:,:,1]) ## or saturation
	y = np.array(transform[:,:,0]) ##intensity or value
	return y,Pb,Pr

def inverse_transform(Kr,Kb,Type,Im):
	if(Type == 0):
		T= Color_Matrix_YUV(Kr,Kb)
	else:
		T= Color_Matrix_BT(Kr,Kb)
	T_inv = np.linalg.inv(T)
	transform = t(pixels,T_inv)
	Pr = transform[:,:,2] ## or Hue
	Pb = transform[:,:,1] ## or saturation
	y = transform[:,:,0] ##intensity or value
	return y,Pb,Pr

def show_color_transform(Im, y,Pb,Pr):
	output = [Im, y, Pb, Pr]
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

