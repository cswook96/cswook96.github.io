---
layout: post
title: "텐서플로 Model CheckPoint"
---



```python
#모듈 임포트
import tensorflow as tf
```

## 데이터 로드,전처리


```python
(x_train,y_train),(x_valid,y_valid) = tf.keras.datasets.mnist.load_data()

x_train = x_train[...,tf.newaxis]/255.0
x_valid = x_valid[...,tf.newaxis]/255.0

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1000).batch(128)
valid_data = tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).batch(128)
```

## 모델링


```python
class MyModel(tf.keras.models.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.cnn1 = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(2,2)
        self.cnn2 = tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D(2,2)
        self.cnn3 = tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.Dense1 = tf.keras.layers.Dense(32,activation='relu')
        self.output_ = tf.keras.layers.Dense(10,activation='softmax')
    
    def call(self,input_):
        x = self.cnn1(input_)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.maxpool2(x)
        x = self.cnn3(x)
        x = self.flatten(x)
        x = self.Dense1(x)
        x = self.output_(x)
        return x
```


```python
model = MyModel()

input_ = tf.keras.layers.Input(shape = (28,28,1))
model(input_)

model.summary()
```

    Model: "my_model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              multiple                  320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) multiple                  0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            multiple                  18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 multiple                  0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            multiple                  36928     
    _________________________________________________________________
    flatten (Flatten)            multiple                  0         
    _________________________________________________________________
    dense (Dense)                multiple                  100384    
    _________________________________________________________________
    dense_1 (Dense)              multiple                  330       
    =================================================================
    Total params: 156,458
    Trainable params: 156,458
    Non-trainable params: 0
    _________________________________________________________________


## CheckPoint


```python
checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True, # 가중치만 저장
                                                save_best_only=True,    # 기준값이 개선 되었을때만 저장
                                                monitor='val_loss',     # 기준값: val_loss
                                                verbose=1)
```

## 컴파일,학습


```python
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['acc'])
```


```python
model.fit(train_data,
         validation_data = (valid_data),
         epochs = 10,
          callbacks=[checkpoint]
         )
```

    Epoch 1/10
    469/469 [==============================] - 24s 50ms/step - loss: 0.5619 - acc: 0.8215 - val_loss: 0.0621 - val_acc: 0.9800
    
    Epoch 00001: val_loss improved from inf to 0.06214, saving model to my_checkpoint.ckpt
    Epoch 2/10
    469/469 [==============================] - 22s 46ms/step - loss: 0.0602 - acc: 0.9821 - val_loss: 0.0519 - val_acc: 0.9833
    
    Epoch 00002: val_loss improved from 0.06214 to 0.05194, saving model to my_checkpoint.ckpt
    Epoch 3/10
    469/469 [==============================] - 21s 46ms/step - loss: 0.0419 - acc: 0.9875 - val_loss: 0.0352 - val_acc: 0.9879
    
    Epoch 00003: val_loss improved from 0.05194 to 0.03525, saving model to my_checkpoint.ckpt
    Epoch 4/10
    469/469 [==============================] - 22s 46ms/step - loss: 0.0294 - acc: 0.9908 - val_loss: 0.0306 - val_acc: 0.9899
    
    Epoch 00004: val_loss improved from 0.03525 to 0.03059, saving model to my_checkpoint.ckpt
    Epoch 5/10
    469/469 [==============================] - 22s 47ms/step - loss: 0.0234 - acc: 0.9926 - val_loss: 0.0281 - val_acc: 0.9897
    
    Epoch 00005: val_loss improved from 0.03059 to 0.02808, saving model to my_checkpoint.ckpt
    Epoch 6/10
    469/469 [==============================] - 23s 49ms/step - loss: 0.0180 - acc: 0.9942 - val_loss: 0.0298 - val_acc: 0.9900
    
    Epoch 00006: val_loss did not improve from 0.02808
    Epoch 7/10
    469/469 [==============================] - 22s 47ms/step - loss: 0.0141 - acc: 0.9951 - val_loss: 0.0432 - val_acc: 0.9875
    
    Epoch 00007: val_loss did not improve from 0.02808
    Epoch 8/10
    469/469 [==============================] - 22s 47ms/step - loss: 0.0136 - acc: 0.9952 - val_loss: 0.0397 - val_acc: 0.9888
    
    Epoch 00008: val_loss did not improve from 0.02808
    Epoch 9/10
    469/469 [==============================] - 22s 46ms/step - loss: 0.0122 - acc: 0.9963 - val_loss: 0.0386 - val_acc: 0.9889
    
    Epoch 00009: val_loss did not improve from 0.02808
    Epoch 10/10
    469/469 [==============================] - 21s 46ms/step - loss: 0.0107 - acc: 0.9966 - val_loss: 0.0390 - val_acc: 0.9900
    
    Epoch 00010: val_loss did not improve from 0.02808





    <tensorflow.python.keras.callbacks.History at 0x18efe95a250>



## 가중치 적용


```python
model.load_weights(checkpoint_path)
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x18e82b9a850>


