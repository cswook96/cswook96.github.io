---
layout: post
title: "텐서플로 Sequential API를 활용한 모델링"
---

```python
# 모듈 임포트
import tensorflow as tf
import tensorflow_datasets as tfds
```

## 데이터(cifar10) 로드,전처리


```python
train_datasets = tfds.load('cifar10',split='train')
valid_datasets = tfds.load('cifar10',split='test')

def preprocessing(data):
    image = tf.cast(data['image'],dtype=tf.float32)/255.0
    label = data['label']
    return image,label

BATCH_SIZE = 128
train_data = train_datasets.map(preprocessing).shuffle(1000).batch(BATCH_SIZE)
valid_data = valid_datasets.map(preprocessing).batch(BATCH_SIZE)
```

## Sequential API 를 활용한 모델링


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(32,32,3)), # 32x32x32
    tf.keras.layers.MaxPool2D(2,2),                                                          # 16x16x32
    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),                       # 16x16x64
    tf.keras.layers.MaxPool2D(2,2),                                                          # 8x8x64
    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),                       # 8x8x64
    tf.keras.layers.MaxPool2D(2,2),                                                          # 4x4x64
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
    
])
```


```python
# model summary(요약)
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 32)        896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 16, 16, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 8, 8, 64)          36928     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1024)              0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                32800     
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                330       
    =================================================================
    Total params: 89,450
    Trainable params: 89,450
    Non-trainable params: 0
    _________________________________________________________________



```python
#model compile
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
```


```python
#model fit( 학습)
model.fit(train_data,
         validation_data=(valid_data),
         epochs = 10)
```

    Epoch 1/10
    391/391 [==============================] - 29s 73ms/step - loss: 1.3004 - acc: 0.5330 - val_loss: 1.2550 - val_acc: 0.5476
    Epoch 2/10
    391/391 [==============================] - 27s 69ms/step - loss: 1.1447 - acc: 0.5941 - val_loss: 1.1183 - val_acc: 0.6069
    Epoch 3/10
    391/391 [==============================] - 27s 68ms/step - loss: 1.0367 - acc: 0.6376 - val_loss: 1.0300 - val_acc: 0.6392
    Epoch 4/10
    391/391 [==============================] - 28s 71ms/step - loss: 0.9580 - acc: 0.6641 - val_loss: 0.9709 - val_acc: 0.6673
    Epoch 5/10
    391/391 [==============================] - 29s 75ms/step - loss: 0.8885 - acc: 0.6907 - val_loss: 0.9372 - val_acc: 0.6753
    Epoch 6/10
    391/391 [==============================] - 27s 70ms/step - loss: 0.8447 - acc: 0.7054 - val_loss: 0.9297 - val_acc: 0.6803
    Epoch 7/10
    391/391 [==============================] - 29s 74ms/step - loss: 0.8060 - acc: 0.7198 - val_loss: 0.8706 - val_acc: 0.7042
    Epoch 8/10
    391/391 [==============================] - 27s 68ms/step - loss: 0.7612 - acc: 0.7350 - val_loss: 0.8435 - val_acc: 0.7131
    Epoch 9/10
    391/391 [==============================] - 27s 69ms/step - loss: 0.7385 - acc: 0.7428 - val_loss: 0.8785 - val_acc: 0.7041
    Epoch 10/10
    391/391 [==============================] - 27s 69ms/step - loss: 0.7092 - acc: 0.7521 - val_loss: 0.8406 - val_acc: 0.7134





    <tensorflow.python.keras.callbacks.History at 0x1c940f84640>
