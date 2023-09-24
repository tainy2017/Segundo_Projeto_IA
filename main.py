#pip install numpy
from matplotlib import pyplot as plt

#import matplotlib.pyplot as plt
import numpy as np
import cv2
#from tensorflow.keras.models import load_model
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from imutils.contours import sort_contours
import imutils

rede_neural = load_model('rede_neural/rede_neural')
rede_neural.summary()

img = cv2.imread('imagens/teste-manuscrito01.jpg')
plt.imshow(img)
plt.show()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()

