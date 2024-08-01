import os,sys 
import numpy as np 
import cv2 


inp = 'output.txt'
data = np.loadtxt(inp).astype(np.float32)
print(data.shape)

out_save = cv2.applyColorMap(cv2.convertScaleAbs(data*200), cv2.COLORMAP_TURBO)
cv2.imwrite('result.png', out_save)
