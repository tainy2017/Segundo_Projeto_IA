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

desfoque = cv2.GaussianBlur(gray, (3,3), 0)
plt.imshow(desfoque)
plt.show()

adapt_media = cv2.adaptiveThreshold(desfoque, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
plt.imshow(adapt_media)
plt.show()

inv = 255 - adapt_media
plt.imshow(adapt_media)
plt.show()

dilatado = cv2.dilate(inv, np.ones((3,3)))
plt.imshow(dilatado)
plt.show()

dilatado = cv2.dilate(inv, np.ones((3,3)))
plt.imshow(dilatado)
plt.show()

bordas = cv2.Canny(desfoque, 40, 150)
plt.imshow(bordas)
plt.show()

dilatado = cv2.dilate(bordas, np.ones((3,3)))
plt.imshow(dilatado)
plt.show()

def encontrar_contornos(img):
  conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  conts = imutils.grab_contours(conts)
  conts = sort_contours(conts, method='left-to-right')[0]
  return conts

conts = encontrar_contornos(dilatado.copy())

l_min, l_max = 4, 160
a_min, a_max = 14, 140

caracteres = []
img_cp = img.copy()
for c in conts:
  #print(c)
  (x, y, w, h) = cv2.boundingRect(c)
  #print(x, y, w, h)
  if (w >= l_min and w <= l_max) and (h >= a_min and h <= a_max):
    roi = gray[y:y+ h, x:x + w]
    #cv2_imshow(roi)
    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    plt.imshow(thresh)
    cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 100, 0), 2)
plt.imshow(img_cp)

