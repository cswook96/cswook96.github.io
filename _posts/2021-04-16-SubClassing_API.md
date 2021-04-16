```python
#모듈 임포트
import tensorflow as tf
import numpy as np
```

## 데이터 로드,전처리(mnist)


```python
(x_train,y_train),(x_valid,y_valid) = tf.keras.datasets.mnist.load_data()

x_train = x_train[...,tf.newaxis]/255.0
x_valid = x_valid[...,tf.newaxis]/255.0
```


```python
#데이터 확인
print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}, x_valid.shape: {x_valid.shape}, y_valid: {y_valid.shape}')
print(f'x_train.min: {np.min(x_train)}, x_train.max: {np.max(x_train)}, x_valid.min: {np.min(np.min(x_valid))}, x_valid.max: {np.min(np.max(x_valid))}')
```

    x_train.shape: (60000, 28, 28, 1), y_train.shape: (60000,), x_valid.shape: (10000, 28, 28, 1), y_valid: (10000,)
    x_train.min: 0.0, x_train.max: 1.0, x_valid.min: 0.0, x_valid.max: 1.0
    

## Sub Classsing API로 모델링


```python
#모델링
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
# 모델 선언
model = MyModel()

input_ = tf.keras.layers.Input(shape = (28,28,1))
model(input_)
```




    <KerasTensor: shape=(None, 10) dtype=float32 (created by layer 'my_model_5')>




```python
#model summary
model.summary()
```

    Model: "my_model_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_15 (Conv2D)           multiple                  320       
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling multiple                  0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           multiple                  18496     
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling multiple                  0         
    _________________________________________________________________
    conv2d_17 (Conv2D)           multiple                  36928     
    _________________________________________________________________
    flatten_5 (Flatten)          multiple                  0         
    _________________________________________________________________
    dense_10 (Dense)             multiple                  100384    
    _________________________________________________________________
    dense_11 (Dense)             multiple                  330       
    =================================================================
    Total params: 156,458
    Trainable params: 156,458
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#model compile
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['acc'])
```


```python
model.fit(x_train,y_train,
         validation_data = (x_valid,y_valid),
         epochs = 10)
```

    Epoch 1/10
    1875/1875 [==============================] - 24s 12ms/step - loss: 0.2904 - acc: 0.9095 - val_loss: 0.0379 - val_acc: 0.9878
    Epoch 2/10
    1875/1875 [==============================] - 25s 13ms/step - loss: 0.0419 - acc: 0.9866 - val_loss: 0.0406 - val_acc: 0.9860
    Epoch 3/10
    1875/1875 [==============================] - 25s 13ms/step - loss: 0.0311 - acc: 0.9902 - val_loss: 0.0309 - val_acc: 0.9895
    Epoch 4/10
    1875/1875 [==============================] - 23s 12ms/step - loss: 0.0218 - acc: 0.9926 - val_loss: 0.0248 - val_acc: 0.9918
    Epoch 5/10
    1875/1875 [==============================] - 23s 12ms/step - loss: 0.0174 - acc: 0.9945 - val_loss: 0.0342 - val_acc: 0.9901
    Epoch 6/10
    1875/1875 [==============================] - 24s 13ms/step - loss: 0.0128 - acc: 0.9958 - val_loss: 0.0285 - val_acc: 0.9924
    Epoch 7/10
    1875/1875 [==============================] - 23s 12ms/step - loss: 0.0097 - acc: 0.9971 - val_loss: 0.0323 - val_acc: 0.9899
    Epoch 8/10
    1875/1875 [==============================] - 23s 12ms/step - loss: 0.0091 - acc: 0.9970 - val_loss: 0.0289 - val_acc: 0.9913
    Epoch 9/10
    1875/1875 [==============================] - 24s 13ms/step - loss: 0.0092 - acc: 0.9969 - val_loss: 0.0300 - val_acc: 0.9922
    Epoch 10/10
    1875/1875 [==============================] - 24s 13ms/step - loss: 0.0072 - acc: 0.9978 - val_loss: 0.0344 - val_acc: 0.9909
    




    <tensorflow.python.keras.callbacks.History at 0x1d13d715a90>


