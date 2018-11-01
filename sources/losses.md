
## 손실 함수 사용법

손실 함수(목표 함수, 최적화 점수 함수)는 모델 컴파일에 꼭 필요한 두 매개변수 중 하나이다.

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

```python
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

기존 손실 함수의 이름을 줄 수도 있고 각 데이터 포인트에 대해 스칼라를 반환하고 다음 두 인자를 받는 텐서플로우/테아노 심볼 함수를 줄 수도 있다.

- __y_true__: 맞는 레이블. 텐서플로우/테아노 텐서.
- __y_pred__: 예측. y_true와 형태가 같은 텐서플로우/테아노 텐서.

실제로 최적화되는 목표는 모든 데이터 포인트에 대한 출력 배열의 평균이다.

그런 함수들의 예 몇 가지를 [losses 소스](https://github.com/keras-team/keras/blob/master/keras/losses.py)에서 확인할 수 있다.

## 사용 가능한 손실 함수

### mean_squared_error


```python
keras.losses.mean_squared_error(y_true, y_pred)
```

----

### mean_absolute_error


```python
keras.losses.mean_absolute_error(y_true, y_pred)
```

----

### mean_absolute_percentage_error


```python
keras.losses.mean_absolute_percentage_error(y_true, y_pred)
```

----

### mean_squared_logarithmic_error


```python
keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```

----

### squared_hinge


```python
keras.losses.squared_hinge(y_true, y_pred)
```

----

### hinge


```python
keras.losses.hinge(y_true, y_pred)
```

----

### categorical_hinge


```python
keras.losses.categorical_hinge(y_true, y_pred)
```

----

### logcosh


```python
keras.losses.logcosh(y_true, y_pred)
```


예측 오류의 쌍곡코사인의 로그.

`log(cosh(x))`는 작은 `x`에서는 `(x ** 2) / 2`와 거의 같고
큰 `x`에서는 `abs(x) - log(2)`에 가깝다. 즉 'logcosh'는
거의 평균 제곱 오차처럼 동작하되 가끔씩 있는 크게 틀린 예측에
그리 강하게 영향을 받지 않는다.

__인자__

- __y_true__: 맞는 목표들의 텐서.
- __y_pred__: 예측한 목표들의 텐서.

__반환__

표본별로 스칼라 손실 항목을 하나씩 담은 텐서.

----

### categorical_crossentropy


```python
keras.losses.categorical_crossentropy(y_true, y_pred)
```

----

### sparse_categorical_crossentropy


```python
keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
```

----

### binary_crossentropy


```python
keras.losses.binary_crossentropy(y_true, y_pred)
```

----

### kullback_leibler_divergence


```python
keras.losses.kullback_leibler_divergence(y_true, y_pred)
```

----

### poisson


```python
keras.losses.poisson(y_true, y_pred)
```

----

### cosine_proximity


```python
keras.losses.cosine_proximity(y_true, y_pred)
```


----

**주의**: `categorical_crossentropy` 손실을 쓸 때는 목표가 범주 형태여야 한다. (가령 10가지 유형이 있다면 각 표본의 목표는 표본의 유형에 대응하는 위치에서만 1이고 나머지는 모두 0인 10차원 벡터여야 한다.) *정수 목표*를 *범주 목표*로 변환하려면 케라스의 유틸리티 `to_categorical`을 쓸 수 있다.

```python
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```
