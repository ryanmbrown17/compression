import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_decomp_lev_1(s1,w11,w12,w13):
	output = [s1,w11,w12,w12]
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.axis('off')
		plt.imshow(output[i], cmap = 'gray')    
	plt.show()
