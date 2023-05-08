import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

##Utility


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

def normalize(X):
	return X / X.max()

###Actual Functions

##Takes an Image and returns Intensity(values), Cb, and Cr
## Image value is given by pixels, Kr and Kb represent the values of the specific color transform, Type if 0 means The color matrix where as 1 is the BT
def transform(Kr, Kb, Type,Im):
	if(Type == 0):
		T= Color_Matrix_YUV(Kr,Kb)
	else:
		T= Color_Matrix_BT(Kr,Kb)
	pixels = np.array(Im)[:,:,0:3]
	transform = normalize(np.matmul(pixels,T))
	return transform[:,:,0],transform[:,:,1],transform[:,:,2] 

def inverse_transform(Kr,Kb,Type,Im):
	if(Type == 0):
		T= Color_Matrix_YUV(Kr,Kb)
	else:
		T= Color_Matrix_BT(Kr,Kb)
	T_inv = np.linalg.inv(T)
	pixels = np.array(Im)[:,:,0:3]
	transform = np.matmul(pixels,T_inv)
	return transform[:,:,0],transform[:,:,1],transform[:,:,2] 
def show_color_transform(Im, y,Pb,Pr):
	output = [Im, Pr, Pb, y]
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


def show_RGB(Im, R,G,B):
	output = [Im, R, G, B]
	title = ['Image','Red','Green','Blue']
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.axis('off')
		plt.title(title[i])
		if i == 0:
			plt.imshow(output[i])
		if i == 1:
			plt.imshow(output[i], cmap = 'Reds')    
		if i == 2:
			plt.imshow(output[i], cmap = 'Greens')
		if i == 3:
			plt.imshow(output[i], cmap = 'Blues')        
	plt.show()
