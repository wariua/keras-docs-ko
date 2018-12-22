## 초기화 사용

초기화는 케라스 층들의 최초 임의 가중치를 설정하는 방식을 지정한다.

층에 초기화 방식을 전달하는 데 쓰는 키워드 인자가 층에 따라 다를 수 있다. 일반적으로는 `kernel_initializer`와 `bias_initializer`다.

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## 사용 가능한 초기화

`keras.initializers` 모듈에 포함된 다음 내장 초기화 방식들을 사용할 수 있다.

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L229)</span>
### Orthogonal

```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

난수 직교 행렬을 생성하는 초기화.

__인자__

- __gain__: 직교 행렬에 적용할 곱셈 계수.
- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

__참고 자료__

Saxe 외, http://arxiv.org/abs/1312.6120

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L266)</span>
### Identity

```python
keras.initializers.Identity(gain=1.0)
```

단위 행렬을 생성하는 초기화.

2차원 정방 행렬에만 사용.

__인자__

- __gain__: 단위 행렬에 적용할 곱셈 계수.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L14)</span>
### Initializer

```python
keras.initializers.Initializer()
```

초기화 기반 클래스. 모든 초기화 방식들이 이 클래스를 상속한다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L33)</span>
### Zeros

```python
keras.initializers.Zeros()
```

0으로 초기화된 텐서들을 생성하는 초기화.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L41)</span>
### Ones

```python
keras.initializers.Ones()
```

1로 초기화된 텐서들을 생성하는 초기화.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L49)</span>
### Constant

```python
keras.initializers.Constant(value=0)
```

상수 값으로 초기화된 텐서들을 생성하는 초기화.

__인자__

- __value__: float. 생성 텐서들의 값.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L66)</span>
### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```

정규 분포로 텐서들을 생성하는 초기화.

__인자__

- __mean__: 파이썬 스칼라 또는 스칼라 텐서. 생성할 난수 값의 평균.
- __stddev__: 파이썬 스칼라 또는 스칼라 텐서. 생성할 난수 값의 표준 편차.
- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L94)</span>
### RandomUniform

```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

균일 분포로 텐서들을 생성하는 초기화.

__인자__

- __minval__: 파이썬 스칼라 또는 스칼라 텐서. 생성할 난수 값 범위의 하한.
- __maxval__: 파이썬 스칼라 또는 스칼라 텐서. 생성할 난수 값 범위의 상한.
  float 타입에는 기본값이 1.
- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L122)</span>
### TruncatedNormal

```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

절단 정규 분포를 생성하는 초기화.

이 값은 `RandomNormal`의 값과 비슷하되
평균에서 표준 편차 두 배 넘게 떨어진 값은
버리고 다시 뽑는다. 신경망 가중치 및 필터에
권장하는 초기화 방식이다.

__인자__

- __mean__: 파이썬 스칼라 또는 스칼라 텐서. 생성할 난수 값의 평균.
- __stddev__: 파이썬 스칼라 또는 스칼라 텐서. 생성할 난수 값의 표준 편차.
- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L155)</span>
### VarianceScaling

```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

가중치 형태에 따라 값 크기를 조정할 수 있는 초기화.

`distribution="normal"`이면 0이 중심이고 `stddev = sqrt(scale / n)`인
절단 정규 분포에서 표본을 뽑는다. 여기서 n은

- mode = "fan_in"이면 가중치 텐서의 입력 유닛 수
- mode = "fan_out"이면 출력 유닛 수
- mode = "fan_avg"이면 입력 및 출력 유닛 수의 평균

`distribution='uniform"`이면
[-limit, limit] 범위 균일 분포에서 표본을 뽑는다.
`limit = sqrt(3 * scale / n)`이다.

__인자__

- __scale__: 크기 인자 (양수 float).
- __mode__: "fan_in", "fan_out", "fan_avg" 중 하나.
- __distribution__: 사용할 난수 분포. "normal" 또는 "uniform".
- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

__예외__

- __ValueError__: "scale", "mode", "distribution" 인자 값이
  유효하지 않은 경우.

----

### lecun_uniform


```python
keras.initializers.lecun_uniform(seed=None)
```


LeCun 균일 초기화.

[-limit, limit] 범위 균일 분포에서 표본을 뽑는다.
여기서 `limit`은 `sqrt(3 / fan_in)`인데
`fan_in`은 가중치 텐서의 입력 유닛 수이다.

__인자__

- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

__반환__

initializer.

__참고 자료__

LeCun 98, Efficient Backprop,
- http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

----

### glorot_normal


```python
keras.initializers.glorot_normal(seed=None)
```


Glorot 정규 초기화. Xavier 정규 초기화라고도 함.

0이 중심이고 `stddev = sqrt(2 / (fan_in + fan_out))`인
절단 정규 분포에서 표본을 뽑는다.
여기서 `fan_in`은 가중치 텐서의 입력 유닛 수이고
`fan_out`은 가중치 텐서의 출력 유닛 수이다.

__인자__

- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

__반환__

initializer.

__참고 자료__

Glorot & Bengio, AISTATS 2010
- http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

----

### glorot_uniform


```python
keras.initializers.glorot_uniform(seed=None)
```


Glorot 균일 초기화. Xavier 균일 초기화라고도 함.

[-limit, limit] 범위 균일 분포에서 표본을 뽑는다.
여기서 `limit`은 `sqrt(6 / (fan_in + fan_out))`인데
`fan_in`은 가중치 텐서의 입력 유닛 수이고
`fan_out`은 가중치 텐서의 출력 유닛 수이다.

__인자__

- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

__반환__

initializer.

__참고 자료__

Glorot & Bengio, AISTATS 2010
- http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

----

### he_normal


```python
keras.initializers.he_normal(seed=None)
```


He 정규 초기화.

0이 중심이고 `stddev = sqrt(2 / fan_in)`인
절단 정규 분포에서 표본을 뽑는다.
여기서 `fan_in`은 가중치 텐서의 입력 유닛 수이다.

__인자__

- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

__반환__

initializer.

__참고 자료__

He 외, http://arxiv.org/abs/1502.01852

----

### lecun_normal


```python
keras.initializers.lecun_normal(seed=None)
```


LeCun 정규 초기화.

0이 중심이고 `stddev = sqrt(1 / fan_in)`인
절단 정규 분포에서 표본을 뽑는다.
여기서 `fan_in`은 가중치 텐서의 입력 유닛 수이다.

__인자__

- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

__반환__

initializer.

__참고 자료__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

----

### he_uniform


```python
keras.initializers.he_uniform(seed=None)
```


He 균일 분산 스케일링 초기화.

[-limit, limit] 범위 균일 분포에서 표본을 뽑는다.
여기서 `limit`은 `sqrt(6 / fan_in)`인데
`fan_in`은 가중치 텐서의 입력 유닛 수이다.

__인자__

- __seed__: 파이썬 정수. 난수 생성기 시드로 사용.

__반환__

initializer.

__참고 자료__

He 외, http://arxiv.org/abs/1502.01852



초기화 방식을 (위의 가용 초기화 방식들 중 하나와 일치하는) 문자열로 전달할 수도 있고 호출 가능 객체로 전달할 수도 있다.

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# 이렇게도 가능하다. 매개변수 기본값을 쓴다.
model.add(Dense(64, kernel_initializer='random_normal'))
```


## 자체 초기화 사용하기

자체적인 호출 가능 객체를 전달하려는 경우에는 그 객체가 `shape`(초기화할 변수의 형태)와 `dtype`(생성 값의 dtype)을 인자로 받아야 한다.

```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```
