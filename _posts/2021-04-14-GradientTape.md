---
layout: post
title: "텐서플로 GradientTape 커스텀 학습"
---



```python
# 모듈 임포트
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
```

## 데이터셋 로드,전처리(cifar 10)


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


```python
# 데이터 확인
for image,label in train_data.take(1):
    print(f'image_min: {np.min(image)}, image_max: {np.max(image)}')
    print(f'image_shape: {image[0,:,:,:].shape}')
```

    image_min: 0.0, image_max: 1.0
    image_shape: (32, 32, 3)


## 모델링(Functional API)


```python
input_ = tf.keras.layers.Input(shape=((32,32,3)))

x = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu')(input_)
x = tf.keras.layers.MaxPool2D(2,2)(x)
x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
x = tf.keras.layers.MaxPool2D(2,2)(x)
x = tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32,activation='relu')(x)
x = tf.keras.layers.Dense(10,activation='softmax')(x)

model = tf.keras.models.Model(input_,x)
```


```python
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 32, 32, 3)]       0         
    _________________________________________________________________
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
    flatten (Flatten)            (None, 4096)              0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                131104    
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                330       
    =================================================================
    Total params: 187,754
    Trainable params: 187,754
    Non-trainable params: 0
    _________________________________________________________________


## GradientTape 커스텀 학습


```python
#loss function,optimizer 설정
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
```


```python
# train_loss,train_acc,valid_loss,valid_acc 설정
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_acc')
```


```python
#학습 단계
@tf.function
def train_step(image,label):
    with tf.GradientTape() as tape:
        prediction = model(image,training=True)
        loss = loss_function(label,prediction)
    
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    
    train_loss(loss)
    train_acc(label,prediction)
```


```python
#검증 단계
@tf.function
def valid_step(image,label):
    prediction = model(image,training=False)
    loss = loss_function(label,prediction)
    
    valid_loss(loss)
    valid_acc(label,prediction)
```


```python
EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    valid_loss.reset_states()
    valid_acc.reset_states()
    
    for image,label in train_data:
        train_step(image,label)			# 학습
        
    for image,label in valid_data:
        valid_step(image,label)			# 검증
        
    print(f'epoch: {epoch+1}, loss: {train_loss.result()}, acc: {train_acc.result()}, val_loss: {valid_loss.result()}, val_acc: {valid_acc.result()}')
```

    epoch: 1, loss: 1.6027292013168335, acc: 0.41449999809265137, val_loss: 1.3001593351364136, val_acc: 0.5309000015258789
    epoch: 2, loss: 1.1894820928573608, acc: 0.5776000022888184, val_loss: 1.0690418481826782, val_acc: 0.6276000142097473
    epoch: 3, loss: 1.0045056343078613, acc: 0.6476399898529053, val_loss: 0.9490131139755249, val_acc: 0.666700005531311
    epoch: 4, loss: 0.8914103507995605, acc: 0.6885200142860413, val_loss: 0.9108906984329224, val_acc: 0.6766999959945679
    epoch: 5, loss: 0.8084282875061035, acc: 0.7186800241470337, val_loss: 0.8765252232551575, val_acc: 0.6951000094413757
    epoch: 6, loss: 0.7497376203536987, acc: 0.7407799959182739, val_loss: 0.8219699859619141, val_acc: 0.7143999934196472
    epoch: 7, loss: 0.703594446182251, acc: 0.7564600110054016, val_loss: 0.830111563205719, val_acc: 0.7117999792098999
    epoch: 8, loss: 0.6625998616218567, acc: 0.7717000246047974, val_loss: 0.8066766262054443, val_acc: 0.7217000126838684
    epoch: 9, loss: 0.6258422136306763, acc: 0.7823799848556519, val_loss: 0.7963829040527344, val_acc: 0.7263000011444092
    epoch: 10, loss: 0.580909252166748, acc: 0.7989799976348877, val_loss: 0.7949218153953552, val_acc: 0.7264999747276306

