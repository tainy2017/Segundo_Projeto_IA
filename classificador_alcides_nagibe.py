import tensorflow as tf
#print("Versão do TensorFlow:", tf.__version__)

import keras as K
#print("Versão do Keras:", K.__version__)

# Imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy # This is new!
from keras.preprocessing.image import ImageDataGenerator
import os

current_dir = os.path.abspath(os.getcwd())

# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adicionando a Segunda Camada de Convolução
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Passo 3 - Flattening
classifier.add(Flatten())
# Passo 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compilando a rede
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

folder = "\\alcides_and_nagibe_dataset"
train_folder = current_dir + folder + "\\train"
val_folder = current_dir + folder + "\\validation"
test_folder = current_dir + folder + "\\test"

# Pré-processamento das imagens de treino e validação
training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory(val_folder,
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

# specify the path where you want to save the model
filepath = current_dir + "\\best_model.hdf5"

# initialize the ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

classifier.save(current_dir + "\\best_model.hdf5")

callbacks = [
    ModelCheckpoint(
        filepath="model3.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

# Executando o treinamento (esse processo pode levar bastante tempo, dependendo do seu computador)
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 5,
                         validation_data = validation_set,
                         validation_steps = 2000)

# Primeira Imagem
import numpy as np
from keras.preprocessing import image
from tensorflow import keras
model = keras.models.load_model(current_dir + "best_model.hdf5")
test_image = image.load_img('alcides_and_nagibe_dataset/test/alcides/alcides.1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = model.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Nagibe'
else:
    prediction = 'Alcides'

# Previsão da primeira imagem
prediction

# Segunda Imagem
test_image = image.load_img('alcides_and_nagibe_dataset/test/alcides/alcides.10.JPG', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Nagibe'
else:
    prediction = 'Alcides'

# Previsão da segunda imagem
prediction

# Terceira Imagem
test_image = image.load_img('alcides_and_nagibe_dataset/test/nagibe/nagibe2.PNG', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Alcides'
else:
    prediction = 'Nagibe'

# Previsão da segunda imagem
prediction