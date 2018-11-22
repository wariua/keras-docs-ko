## 레귤러라이저 사용

레귤러라이저를 통해 최적화 중에 층 매개변수나 층 활성에 패널티를 줄 수 있다. 망에서 최적화하는 손실 함수에 그 패널티가 산입된다.

패널티는 층 단위로 적용된다. 정확한 API는 층에 따라 다르겠지만 `Dense`, `Conv1D`, `Conv2D`, `Conv3D` 층에는 공통된 API가 있다.

그 층들에는 3가지 키워드 인자가 있다.

- `kernel_regularizer`: `keras.regularizers.Regularizer`의 인스턴스
- `bias_regularizer`: `keras.regularizers.Regularizer`의 인스턴스
- `activity_regularizer`: `keras.regularizers.Regularizer`의 인스턴스


## 예시

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 사용 가능한 패널티

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

## 새로운 레귤러라이저 개발하기

가중치 행렬을 받아서 손실 기여 텐서를 반환하는 함수를 레귤러라이저로 쓸 수 있다. 예:

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))
```

또는 객체 지향 방식으로 레귤러라이저를 작성할 수도 있다.
[keras/regularizers.py](https://github.com/keras-team/keras/blob/master/keras/regularizers.py) 모듈에서 예를 볼 수 있다.
