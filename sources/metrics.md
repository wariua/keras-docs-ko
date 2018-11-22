
## 지표 사용

지표는 모델의 성능을 판단하는 데 쓰는 함수이다. 모델을 컴파일 할 때 `metrics` 매개변수로 측정 함수를 제공하게 된다.

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

지표 함수는 [손실 함수](/losses)와 비슷하되 지표 평가 결과가 모델 훈련에 쓰이지 않는다는 점이 다르다.

기존 지표의 이름을 줄 수도 있고 테아노/텐서플로우 심볼 함수([자체 지표](#_5) 참고)의 이름을 줄 수도 있다.

#### 인자
  - __y_true__: 맞는 레이블. 테아노/텐서플로우 텐서.
  - __y_pred__: 예측. y_true와 형태가 같은 테아노/텐서플로우 텐서.

#### 반환
  모든 데이터 포인터에 대한 출력 배열의 평균을 나타내는 텐서 값 하나.

----

## 사용 가능한 지표


### binary_accuracy


```python
keras.metrics.binary_accuracy(y_true, y_pred)
```

----

### categorical_accuracy


```python
keras.metrics.categorical_accuracy(y_true, y_pred)
```

----

### sparse_categorical_accuracy


```python
keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
```

----

### top_k_categorical_accuracy


```python
keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

### sparse_top_k_categorical_accuracy


```python
keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```


----

## 자체 지표

컴파일 단계에서 자체적인 지표를 줄 수 있다.
그 함수는 `(y_true, y_pred)`를 인자로 받아서
텐서 값 하나를 반환해야 할 것이다.

```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
