import numpy as np
from keras.preprocessing import image
from tensorflow import keras
import classificador_alcides_nagibe
import os

current_dir = os.path.abspath(os.getcwd())
model = keras.models.load_model(current_dir + "\\best_model.hdf5")
test_image = image.load_img('alcides_and_nagibe_dataset/test/alcides/alcides.1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = model.predict(test_image)
classificador_alcides_nagibe.training_set.class_indices

if result[0][0] == 1:
    prediction = 'Nagibe'
else:
    prediction = 'Alcides'

print(prediction)

# Segunda Imagem
test_image = image.load_img('alcides_and_nagibe_dataset/test/alcides/alcides.10.JPG', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = model.predict(test_image)
classificador_alcides_nagibe.training_set.class_indices

if result[0][0] == 1:
    prediction = 'Nagibe'
else:
    prediction = 'Alcides'

print(prediction)

# Terceira Imagem
test_image = image.load_img('alcides_and_nagibe_dataset/test/nagibe/nagibe2.PNG', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = model.predict(test_image)
classificador_alcides_nagibe.training_set.class_indices

if result[0][0] == 1:
    prediction = 'Alcides'
else:
    prediction = 'Nagibe'

print(prediction)