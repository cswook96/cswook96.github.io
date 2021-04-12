---
layout: post
title: 'tensorflow_datasets를 이용하여 데이터 로드하기'
---



## [tensorflow datasets 공식 도큐먼트](https://www.tensorflow.org/datasets)

## cifar10 데이터 불러오기,전처리


```python
# 모듈 임포트
import tensorflow as tf
import tensorflow_datasets as tfds
```


```python
#데이터셋 로드
train_datasets = tfds.load('cifar10',split='train')
valid_datasets = tfds.load('cifar10',split='test')
```


```python
#데이터 확인하기
for data in train_datasets.take(3):
    image = data['image']
    label = data['label']
    print(image.shape)
    print(label)
```

    (32, 32, 3)
    tf.Tensor(7, shape=(), dtype=int64)
    (32, 32, 3)
    tf.Tensor(8, shape=(), dtype=int64)
    (32, 32, 3)
    tf.Tensor(4, shape=(), dtype=int64)



```python
#데이터 전처리
def preprocessing(data):
    image = tf.cast(data['image'],dtype=tf.float32)/255.0
    label = data['label']
    return image,label
    
BATCH_SIZE = 128
train_data = train_datasets.map(preprocessing).shuffle(1000).batch(BATCH_SIZE)
valid_data = valid_datasets.map(preprocessing).batch(BATCH_SIZE)
```


```python
#데이터 확인하기
for image,label in train_data.take(1):
    print(image.shape)
    print(label.shape)
```

    (128, 32, 32, 3)
    (128,)

