# 实验5_2

## 1.下载压缩包


```python
!wget --no-check-certificate https://storage.googleapis.com/learning-datasets/rps.zip -O rps.zip
  
!wget --no-check-certificate https://storage.googleapis.com/learning-datasets/rps-test-set.zip -O rps-test-set.zip

```

    --2024-06-15 08:48:35--  https://storage.googleapis.com/learning-datasets/rps.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 142.250.68.123, 142.250.72.155, 142.250.72.187, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|142.250.68.123|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 200682221 (191M) [application/zip]
    Saving to: ‘rps.zip’
    
    rps.zip             100%[===================>] 191.38M  33.4MB/s    in 5.7s    
    
    2024-06-15 08:48:42 (33.4 MB/s) - ‘rps.zip’ saved [200682221/200682221]
    
    --2024-06-15 08:48:42--  https://storage.googleapis.com/learning-datasets/rps-test-set.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 142.251.40.59, 142.250.72.187, 172.217.14.91, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|142.251.40.59|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 29516758 (28M) [application/zip]
    Saving to: ‘rps-test-set.zip’
    
    rps-test-set.zip    100%[===================>]  28.15M  60.9MB/s    in 0.5s    
    
    2024-06-15 08:48:43 (60.9 MB/s) - ‘rps-test-set.zip’ saved [29516758/29516758]


​    

## 2.解压压缩包


```python
import os
import zipfile

local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('data')
zip_ref.close()

local_zip = 'rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('data')
zip_ref.close()

```

## 3.检测数据集的解压结果，打印相关信息。


```python
rock_dir = os.path.join('data/rps/rock')
paper_dir = os.path.join('data/rps/paper')
scissors_dir = os.path.join('data/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

```

    total training rock images: 840
    total training paper images: 840
    total training scissors images: 840
    ['rock05ck01-068.png', 'rock07-k03-118.png', 'rock04-114.png', 'rock01-103.png', 'rock01-043.png', 'rock04-030.png', 'rock04-050.png', 'rock05ck01-018.png', 'rock01-041.png', 'rock04-049.png']
    ['paper06-044.png', 'paper05-081.png', 'paper03-089.png', 'paper02-000.png', 'paper03-008.png', 'paper07-046.png', 'paper02-054.png', 'paper06-008.png', 'paper07-000.png', 'paper05-082.png']
    ['testscissors03-083.png', 'testscissors03-046.png', 'testscissors01-024.png', 'scissors03-092.png', 'scissors03-112.png', 'testscissors03-014.png', 'testscissors02-026.png', 'scissors02-118.png', 'scissors01-085.png', 'testscissors02-119.png']

## 4.各打印两张石头剪刀布训练集图片


```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname)
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname)
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname)
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

```

![image-20240615184202161](.\pic\image-20240615184202161.png)

![image-20240615184233175](.\pic\image-20240615184233175.png)



## 5.调用TensorFlow的[keras](https://so.csdn.net/so/search?q=keras&spm=1001.2101.3001.7020)进行数据模型的训练和评估。


```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "data/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "data/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

```

    2024-06-15 08:48:47.790218: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2024-06-15 08:48:47.790253: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.


    2024-06-15 08:48:55.060966: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2024-06-15 08:48:55.061003: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2024-06-15 08:48:55.061032: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-11eef8): /proc/driver/nvidia/version does not exist
    2024-06-15 08:48:55.061416: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
     2D)                                                             
                                                                     
     conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 6272)              0         
                                                                     
     dropout (Dropout)           (None, 6272)              0         
                                                                     
     dense (Dense)               (None, 512)               3211776   
                                                                     
     dense_1 (Dense)             (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 3,473,475
    Trainable params: 3,473,475
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/25


    2024-06-15 08:48:57.777519: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.
    2024-06-15 08:49:00.372037: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.


     1/20 [>.............................] - ETA: 1:28 - loss: 1.1163 - accuracy: 0.3095
    
    2024-06-15 08:49:00.958813: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.
    2024-06-15 08:49:03.480090: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.


     2/20 [==>...........................] - ETA: 56s - loss: 5.1471 - accuracy: 0.2976 
    
    2024-06-15 08:49:04.103763: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.


    20/20 [==============================] - 63s 3s/step - loss: 1.5067 - accuracy: 0.3532 - val_loss: 1.0771 - val_accuracy: 0.3333
    Epoch 2/25
    20/20 [==============================] - 61s 3s/step - loss: 1.1536 - accuracy: 0.3647 - val_loss: 1.0939 - val_accuracy: 0.3333
    Epoch 3/25
    20/20 [==============================] - 64s 3s/step - loss: 1.0863 - accuracy: 0.4310 - val_loss: 0.7698 - val_accuracy: 0.6586
    Epoch 4/25
    20/20 [==============================] - 61s 3s/step - loss: 1.0754 - accuracy: 0.4345 - val_loss: 1.2169 - val_accuracy: 0.3333
    Epoch 5/25
    20/20 [==============================] - 61s 3s/step - loss: 1.1320 - accuracy: 0.5024 - val_loss: 0.9333 - val_accuracy: 0.6156
    Epoch 6/25
    20/20 [==============================] - 61s 3s/step - loss: 0.8634 - accuracy: 0.5885 - val_loss: 0.4521 - val_accuracy: 0.9113
    Epoch 7/25
    20/20 [==============================] - 62s 3s/step - loss: 0.7423 - accuracy: 0.6897 - val_loss: 0.3075 - val_accuracy: 0.9247
    Epoch 8/25
    20/20 [==============================] - 62s 3s/step - loss: 0.5784 - accuracy: 0.7488 - val_loss: 0.8152 - val_accuracy: 0.6371
    Epoch 9/25
    20/20 [==============================] - 61s 3s/step - loss: 0.5776 - accuracy: 0.7536 - val_loss: 0.2356 - val_accuracy: 0.9973
    Epoch 10/25
    20/20 [==============================] - 61s 3s/step - loss: 0.4806 - accuracy: 0.8107 - val_loss: 0.1566 - val_accuracy: 0.9597
    Epoch 11/25
    20/20 [==============================] - 61s 3s/step - loss: 0.3230 - accuracy: 0.8663 - val_loss: 0.2218 - val_accuracy: 0.9247
    Epoch 12/25
    20/20 [==============================] - 62s 3s/step - loss: 0.2866 - accuracy: 0.8845 - val_loss: 0.0904 - val_accuracy: 0.9570
    Epoch 13/25
    20/20 [==============================] - 62s 3s/step - loss: 0.2134 - accuracy: 0.9250 - val_loss: 0.0242 - val_accuracy: 1.0000
    Epoch 14/25
    20/20 [==============================] - 62s 3s/step - loss: 0.2540 - accuracy: 0.9103 - val_loss: 0.0944 - val_accuracy: 0.9543
    Epoch 15/25
    20/20 [==============================] - 62s 3s/step - loss: 0.1469 - accuracy: 0.9460 - val_loss: 0.0653 - val_accuracy: 0.9597
    Epoch 16/25
    20/20 [==============================] - 63s 3s/step - loss: 0.1733 - accuracy: 0.9401 - val_loss: 0.1733 - val_accuracy: 0.9489
    Epoch 17/25
    20/20 [==============================] - 62s 3s/step - loss: 0.1585 - accuracy: 0.9385 - val_loss: 0.0407 - val_accuracy: 0.9812
    Epoch 18/25
    20/20 [==============================] - 62s 3s/step - loss: 0.1760 - accuracy: 0.9353 - val_loss: 0.0973 - val_accuracy: 0.9677
    Epoch 19/25
    20/20 [==============================] - 62s 3s/step - loss: 0.0840 - accuracy: 0.9698 - val_loss: 0.1174 - val_accuracy: 0.9462
    Epoch 20/25
    20/20 [==============================] - 61s 3s/step - loss: 0.1413 - accuracy: 0.9496 - val_loss: 0.0405 - val_accuracy: 0.9758
    Epoch 21/25
    20/20 [==============================] - 62s 3s/step - loss: 0.1326 - accuracy: 0.9472 - val_loss: 0.0310 - val_accuracy: 0.9839
    Epoch 22/25
    20/20 [==============================] - 62s 3s/step - loss: 0.0873 - accuracy: 0.9714 - val_loss: 0.0211 - val_accuracy: 1.0000
    Epoch 23/25
    20/20 [==============================] - 62s 3s/step - loss: 0.0717 - accuracy: 0.9770 - val_loss: 0.0318 - val_accuracy: 0.9866
    Epoch 24/25
    20/20 [==============================] - 61s 3s/step - loss: 0.1485 - accuracy: 0.9492 - val_loss: 0.0490 - val_accuracy: 0.9785
    Epoch 25/25
    20/20 [==============================] - 62s 3s/step - loss: 0.0582 - accuracy: 0.9813 - val_loss: 0.0832 - val_accuracy: 0.9570



```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

```

<img src=".\pic\image-20240615184249583.png"> 

    <Figure size 640x480 with 0 Axes>

