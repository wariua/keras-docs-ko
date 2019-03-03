# 케라스 백엔드

## "백엔드"란?

케라스는 모델 수준 라이브러리이므로 심층학습 모델 개발을 위한 고수준 구성 요소들을 제공한다. 즉 텐서 곱이나 합성곱 같은 저수준 연산들을 자체적으로 다루지 않는다. 대신 최적화된 전문 텐서 조작 라이브러리에 그 일을 맡기는데, 그 라이브러리가 케라스의 "백엔드 엔진"이 된다. 그런데 케라스에서는 한 가지 텐서 라이브러리를 선정해서 케라스 구현이 그 라이브러리에 묶이게 하기보다는 모듈 방식으로 그 문제를 다루며, 그래서 여러 백엔드 엔진들을 케라스에 매끄럽게 연결할 수 있다.

현재 케라스에서 세 가지 백엔드 구현을 사용할 수 있다. **텐서플로우** 백엔드, **테아노** 백엔드, **CNTK** 백엔드이다.

- [텐서플로우(TensorFlow)](http://www.tensorflow.org/)는 구글에서 개발한 오픈 소스 심볼형 텐서 조작 프레임워크다.
- [테아노(Theano)](http://deeplearning.net/software/theano/)는 몬트리올 대학 LISA 실험실에서 개발한 오픈 소스 심볼형 텐서 조작 프레임워크다.
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/)는 마이크로소프트에서 개발한 오픈 소스 심층학습 툴킷이다.

아마 향후에 백엔드 선택지가 더 늘어날 것이다.

----

## 한 백엔드에서 다른 백엔드로 전환하기

케라스를 한 번이라도 실행했다면 다음 위치에 케라스 설정 파일이 있다.

`$HOME/.keras/keras.json`

없다면 만들어도 된다.

**윈도우 사용자 주의 사항:** `$HOME`을 `%USERPROFILE%`로 바꿔야 한다.

기본 설정 파일 내용은 다음과 같다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

`backend` 필드를 `"theano"`, `"tensorflow"`, `"cntk"` 중 하나로 바꿔 주기만 하면 다음 번 케라스 코드 실행 때 케라스에서 새 설정을 사용하게 된다.

환경 변수 ``KERAS_BACKEND``를 정의할 수도 있다. 그러면 설정 파일에
정의된 값을 무시하게 된다.

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend"
Using TensorFlow backend.
```

----

## keras.json 상세 설명


`keras.json` 설정 파일은 다음 설정을 담고 있다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

`$HOME/.keras/keras.json`을 편집해서 이 설정들을 바꿀 수 있다.

* `image_data_format`: 문자열. `"channels_last"` 또는 `"channels_first"`. 케라스에서 따를 데이터 형식 규약을 지정한다. (`keras.backend.image_data_format()`이 반환하는 값.)
    - 2차원 데이터(가령 이미지)에서 `"channels_last"`면 `(rows, cols, channels)`를 상정하고 `"channels_first"`이면 `(channels, rows, cols)`를 상정한다.
    - 3차원 데이터에서 `"channels_last"`면 `(conv_dim1, conv_dim2, conv_dim3, channels)`를 상정하고 `"channels_first"`는 `(channels, conv_dim1, conv_dim2, conv_dim3)`를 상정한다.
* `epsilon`: 실수. 일부 연산에서 0으로 나누기를 피하기 위해 쓰는 퍼징 상수.
* `floatx`: 문자열. `"float16"` 또는 `"float32"` 또는 `"float64"`. 기본 실수 정밀도.
* `backend`: 문자열. `"tensorflow"` 또는 `"theano"` 또는 `"cntk"`.

----

## 케라스 추상 백엔드를 사용해 코드 작성하기

작성하는 케라스 모듈이 테아노(`th`)와 텐서플로우(`tf`) 모두와 호환되게 하고 싶다면 케라스의 추상 백엔드 API를 통해 작성해야 한다. 여기서 간단히 소개한다.

다음 코드로 백엔드 모듈을 임포트 할 수 있다.
```python
from keras import backend as K
```

아래 코드는 입력 플레이스홀더를 만든다. `tf.placeholder()`이나 `th.tensor.matrix()`, `th.tensor.tensor3()` 등과 동등하다.

```python
inputs = K.placeholder(shape=(2, 4, 5))
# 다음도 가능:
inputs = K.placeholder(shape=(None, 4, 5))
# 다음도 가능:
inputs = K.placeholder(ndim=3)
```

아래 코드는 변수를 만든다. `tf.Variable()`이나 `th.shared()`와 동등하다.

```python
import numpy as np
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# 모두 0인 변수:
var = K.zeros(shape=(3, 4, 5))
# 모두 1:
var = K.ones(shape=(3, 4, 5))
```

대부분의 텐서 연산을 텐서플로우나 테아노에서처럼 할 수 있다.

```python
# 난수로 텐서 초기화
b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # 균일 분포
c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # 가우스 분포
d = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)

# 텐서 연산
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=1)
a = K.softmax(b)
a = K.concatenate([b, c], axis=-1)
# 등등...
```

----

## 백엔드 함수


### is_sparse


```python
keras.backend.is_sparse(tensor)
```


텐서가 희소 텐서인지 여부를 반환한다.

__인자__

- __tensor__: 텐서 인스턴스.

__반환__

bool.

__예시__

```python
>>> from keras import backend as K
>>> a = K.placeholder((2, 2), sparse=False)
>>> print(K.is_sparse(a))
False
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
```

----

### to_dense


```python
keras.backend.to_dense(tensor)
```


희소 텐서를 밀집 텐서로 변환해서 반환한다.

__인자__

- __tensor__: (희소일 수도 있는) 텐서 인스턴스.

__반환__

밀집 텐서.

__예시__

```python
>>> from keras import backend as K
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
>>> c = K.to_dense(b)
>>> print(K.is_sparse(c))
False
```

----

### variable


```python
keras.backend.variable(value, dtype=None, name=None, constraint=None)
```


변수를 만들어서 반환한다.

__인자__

- __value__: Numpy 배열. 텐서의 초깃값.
- __dtype__: 텐서 타입.
- __name__: 선택적. 텐서 이름 문자열.
- __constraint__: 선택적. 최적화 갱신 후
    변수에 적용할 투사 함수.

__반환__

(케라스 메타데이터가 포함된) 변수 인스턴스.

__예시__

```python
>>> from keras import backend as K
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val, dtype='float64', name='example_var')
>>> K.dtype(kvar)
'float64'
>>> print(kvar)
example_var
>>> K.eval(kvar)
array([[ 1.,  2.],
       [ 3.,  4.]])
```

----

### constant


```python
keras.backend.constant(value, dtype=None, shape=None, name=None)
```


상수 텐서를 만든다.

__인자__

- __value__: 상수 값 (또는 리스트).
- __dtype__: 결과 텐서 원소들의 타입.
- __shape__: 선택적. 결과 텐서의 차원들.
- __name__: 선택적. 텐서 이름.

__반환__

상수 텐서.

----

### is_keras_tensor


```python
keras.backend.is_keras_tensor(x)
```


`x`가 케라스 텐서인지 여부를 반환한다.

"케라스 텐서"란 케라스 층(`Layer` 클래스)이나 `Input`에서
반환한 텐서이다.

__인자__

- __x__: 후보 텐서.

__반환__

bool: 인자가 케라스 텐서인지 여부.

__예외__

- __ValueError__: `x`가 심볼릭 텐서가 아닌 경우.

__예시__

```python
>>> from keras import backend as K
>>> from keras.layers import Input, Dense
>>> np_var = numpy.array([1, 2])
>>> K.is_keras_tensor(np_var) # numpy 배열은 심볼릭 텐서가 아니다.
ValueError
>>> k_var = tf.placeholder('float32', shape=(1,1))
>>> K.is_keras_tensor(k_var) # 케라스 밖에서 간접적으로 생성된 변수는 케라스 텐서가 아니다.
False
>>> keras_var = K.variable(np_var)
>>> K.is_keras_tensor(keras_var)  # 케라스 백엔스로 생성한 변수는 케라스 텐서가 아니다.
False
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> K.is_keras_tensor(keras_placeholder)  # 플레이스홀더는 케라스 텐서가 아니다.
False
>>> keras_input = Input([10])
>>> K.is_keras_tensor(keras_input) # Input은 케라스 텐서이다.
True
>>> keras_layer_output = Dense(10)(keras_input)
>>> K.is_keras_tensor(keras_layer_output) # 케라스 층의 출력은 모두 케라스 텐서이다.
True
```

----

### is_tensor


```python
keras.backend.is_tensor(x)
```

----

### placeholder


```python
keras.backend.placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None)
```


플레이스홀더 텐서를 만들어서 반환한다.

__인자__

- __shape__: 플레이스홀더의 형태.
    (정수 튜플이며 `None` 성분 포함 가능.)
- __ndim__: 텐서의 축 개수.
    {`shape`, `ndim`} 중 적어도 하나는 지정해야 한다.
    둘 다 지정하면 `shape`를 쓴다.
- __dtype__: 플레이스홀더 타입.
- __sparse__: bool. 플레이스홀더가 희소 유형이어야 하는지 여부.
- __name__: 선택적. 플레이스홀더 이름 문자열.

__반환__

(케라스 메타데이터가 포함된) 텐서 인스턴스.

__예시__

```python
>>> from keras import backend as K
>>> input_ph = K.placeholder(shape=(2, 4, 5))
>>> input_ph._keras_shape
(2, 4, 5)
>>> input_ph
<tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
```

----

### is_placeholder


```python
keras.backend.is_placeholder(x)
```


`x`가 플레이스홀더인지 여부를 반환한다.

__인자__

- __x__: 플레이스홀더 후보.

__반환__

bool.

----

### shape


```python
keras.backend.shape(x)
```


텐서나 변수의 기호로 나타낸 형태를 반환한다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

기호로 나타낸 형태 (그 자체가 텐서).

__예시__

```python
# 텐서플로우 예시
>>> from keras import backend as K
>>> tf_session = K.get_session()
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> inputs = keras.backend.placeholder(shape=(2, 4, 5))
>>> K.shape(kvar)
<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
>>> K.shape(inputs)
<tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
# 정수 형태 얻기 (또는 K.int_shape(x) 사용 가능)
>>> K.shape(kvar).eval(session=tf_session)
array([2, 2], dtype=int32)
>>> K.shape(inputs).eval(session=tf_session)
array([2, 4, 5], dtype=int32)
```

----

### int_shape


```python
keras.backend.int_shape(x)
```


텐서나 변수의 형태를 정수 또는 None 원소들의 튜플로 반환한다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

정수들의 (또는 None 원소들의) 튜플.

__예시__

```python
>>> from keras import backend as K
>>> inputs = K.placeholder(shape=(2, 4, 5))
>>> K.int_shape(inputs)
(2, 4, 5)
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.int_shape(kvar)
(2, 2)
```

----

### ndim


```python
keras.backend.ndim(x)
```


텐서의 축 개수를 정수로 반환한다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

정수 (스칼라). 축 개수.

__예시__

```python
>>> from keras import backend as K
>>> inputs = K.placeholder(shape=(2, 4, 5))
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.ndim(inputs)
3
>>> K.ndim(kvar)
2
```

----

### dtype


```python
keras.backend.dtype(x)
```


케라스 텐서나 변수의 dtype을 문자열로 반환한다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

문자열. `x`의 dtype.

__예시__

```python
>>> from keras import backend as K
>>> K.dtype(K.placeholder(shape=(2,4,5)))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
'float64'
# 케라스 변수
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
>>> K.dtype(kvar)
'float32_ref'
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.dtype(kvar)
'float32_ref'
```

----

### eval


```python
keras.backend.eval(x)
```


변수 값을 평가한다.

__인자__

- __x__: 변수.

__반환__

Numpy 배열.

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.eval(kvar)
array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)
```

----

### zeros


```python
keras.backend.zeros(shape, dtype=None, name=None)
```


모두 0인 변수를 만들어서 반환한다.

__인자__

- __shape__: 정수들의 튜플. 반환되는 케라스 변수의 형태.
- __dtype__: 문자열. 반환되는 케라스 변수의 타입.
- __name__: 문자열. 반환되는 케라스 변수의 이름.

__반환__

`0.0`으로 채워진 (케라스 메타데이터를 포함한) 변수.
참고로 `shape`가 심볼이면 변수를 반환할 수 없으므로
대신 동적 형태 텐서를 반환하게 된다.

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.zeros((3,4))
>>> K.eval(kvar)
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]], dtype=float32)
```

----

### ones


```python
keras.backend.ones(shape, dtype=None, name=None)
```


모두 1인 변수를 만들어서 반환한다.

__인자__

- __shape__: 정수들의 튜플. 반환되는 케라스 변수의 형태.
- __dtype__: 문자열. 반환되는 케라스 변수의 타입.
- __name__: 문자열. 반환되는 케라스 변수의 이름.

__반환__

`1.0`으로 채워진 케라스 변수.
참고로 `shape`가 심볼이면 변수를 반환할 수 없으므로
대신 동적 형태 텐서를 반환하게 된다.

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.ones((3,4))
>>> K.eval(kvar)
array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]], dtype=float32)
```

----

### eye


```python
keras.backend.eye(size, dtype=None, name=None)
```


단위행렬을 만들어서 반환한다.

__인자__

- __size__: 정수. 행/열 개수.
- __dtype__: 문자열. 반환되는 케라스 변수의 타입.
- __name__: 문자열. 반환되는 케라스 변수의 이름.

__반환__

단위행렬인 케라스 변수.

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.eye(3)
>>> K.eval(kvar)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]], dtype=float32)
```


----

### zeros_like


```python
keras.backend.zeros_like(x, dtype=None, name=None)
```


다른 텐서와 형태가 같은 모두 0인 변수를 만든다.

__인자__

- __x__: 케라스 변수 또는 케라스 텐서.
- __dtype__: 문자열. 반환되는 케라스 변수의 dtype.
     None이면 x의 dtype 사용.
- __name__: 문자열. 생성할 변수의 이름.

__반환__

x와 같은 형태이고 0으로 채워진 케라스 변수.

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_zeros = K.zeros_like(kvar)
>>> K.eval(kvar_zeros)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```

----

### ones_like


```python
keras.backend.ones_like(x, dtype=None, name=None)
```


다른 텐서와 형태가 같은 모두 1인 변수를 만든다.

__인자__

- __x__: 케라스 변수 또는 텐서.
- __dtype__: 문자열. 반환되는 케라스 변수의 dtype.
     None이면 x의 dtype 사용.
- __name__: 문자열. 생성할 변수의 이름.

__반환__

x와 같은 형태이고 1으로 채워진 케라스 변수.

__예시__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_ones = K.ones_like(kvar)
>>> K.eval(kvar_ones)
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]], dtype=float32)
```

----

### identity


```python
keras.backend.identity(x, name=None)
```


입력 텐서와 내용물이 같은 텐서를 반환한다.

__인자__

- __x__: 입력 텐서.
- __name__: 문자열. 생성할 변수의 이름.

__반환__

형태, 타입, 내용물이 같은 텐서.

----

### random_uniform_variable


```python
keras.backend.random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None)
```


균일 분포에서 뽑은 값들로 변수를 만든다.

__인자__

- __shape__: 정수들의 튜플. 반환되는 케라스 변수의 형태.
- __low__: 실수. 출력 구간의 하한.
- __high__: 실수. 출력 구간의 상한.
- __dtype__: 문자열. 반환되는 케라스 변수의 dtype.
- __name__: 문자열. 반환되는 케라스 변수의 이름.
- __seed__: 정수. 난수 시드.

__반환__

뽑은 표본들로 채워진 케라스 변수.

__예시__

```python
# TensorFlow example
>>> kvar = K.random_uniform_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
>>> K.eval(kvar)
array([[ 0.10940075,  0.10047495,  0.476143  ],
       [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
```

----

### random_normal_variable


```python
keras.backend.random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None)
```


정규 분포에서 뽑은 값들로 변수를 만든다.

__인자__

- __shape__: 정수들의 튜플. 반환되는 케라스 변수의 형태.
- __mean__: 실수. 정규 분포의 평균.
- __scale__: 실수. 정규 분포의 표준 편차.
- __dtype__: 문자열. 반환되는 케라스 변수의 dtype.
- __name__: 문자열. 반환되는 케라스 변수의 이름.
- __seed__: 정수. 난수 시드.

__반환__

뽑은 표본들로 채워진 케라스 변수.

__예시__

```python
# TensorFlow example
>>> kvar = K.random_normal_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
>>> K.eval(kvar)
array([[ 1.19591331,  0.68685907, -0.63814116],
       [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
```

----

### count_params


```python
keras.backend.count_params(x)
```


케라스 변수 또는 텐서의 고정 원소 개수를 반환한다.

__인자__

- __x__: 케라스 변수 또는 텐서.

__반환__

정수. `x`의 원소 개수. 즉 배열의 고정 차원들의 곱.

__예시__

```python
>>> kvar = K.zeros((2,3))
>>> K.count_params(kvar)
6
>>> K.eval(kvar)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```

----

### cast


```python
keras.backend.cast(x, dtype)
```


텐서를 다른 dtype으로 캐스팅 해서 반환한다.

케라스 변수도 캐스팅 할 수 있지만 마찬가지로 케라스 텐서를 반환한다.

__인자__

- __x__: 케라스 텐서 (또는 변수).
- __dtype__: 문자열. `'float16'`, `'float32'`, `'float64'` 중 하나.

__반환__

dtype이 `dtype`인 케라스 텐서.

__예시__

```python
>>> from keras import backend as K
>>> input = K.placeholder((2, 3), dtype='float32')
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
# 아래에서 보듯 자체에 동작하지 않는다.
>>> K.cast(input, dtype='float16')
<tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
# 할당을 해 줘야 한다.
>>> input = K.cast(input, dtype='float16')
>>> input
<tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
```

----

### update


```python
keras.backend.update(x, new_x)
```


`x`의 값을 `new_x`로 갱신한다.

__인자__

- __x__: 변수.
- __new_x__: `x`와 형태가 같은 텐서.

__반환__

갱신된 변수 `x`.

----

### update_add


```python
keras.backend.update_add(x, increment)
```


`x`의 값에 `increment`를 더한다.

__인자__

- __x__: 변수.
- __increment__: `x`와 형태가 같은 텐서.

__반환__

갱신된 변수 `x`.

----

### update_sub


```python
keras.backend.update_sub(x, decrement)
```


`x`의 값에서 `decrement`를 뺀다.

__인자__

- __x__: 변수.
- __decrement__: `x`와 형태가 같은 텐서.

__반환__

갱신된 변수 `x`.

----

### moving_average_update


```python
keras.backend.moving_average_update(x, value, momentum)
```


변수의 이동 평균을 계산한다.

__인자__

- __x__: 변수.
- __value__: `x`와 형태가 같은 텐서.
- __momentum__: 이동 평균 모멘텀.

__반환__

변수를 갱신할 연산.

----

### dot


```python
keras.backend.dot(x, y)
```


두 텐서 (및/또는 변수)를 곱해서 *텐서*를 반환한다.

n차원 텐서를 n차원 텐서와 곱할 때
테아노 동작 방식을 따른다.
(가령 `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

`x`와 `y`의 도트곱인 텐서.

__예시__

```python
# 텐서끼리 도트곱
>>> x = K.placeholder(shape=(2, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
```

```python
# 텐서끼리 도트곱
>>> x = K.placeholder(shape=(32, 28, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
```

```python
# 테아노식 동작 예시
>>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
>>> y = K.ones((4, 3, 5))
>>> xy = K.dot(x, y)
>>> K.int_shape(xy)
(2, 4, 5)
```

----

### batch_dot


```python
keras.backend.batch_dot(x, y, axes=None)
```


배치 도트곱.

`batch_dot`을 사용해 `x`와 `y`의 도트곱을 계산한다.
이때 `x`와 `y`는 데이터 배치이다. 즉 `(batch_size, :)`
형태이다.
`batch_dot`은 입력보다 차원이 적은 텐서 내지 변수를
내놓는다. 차원 개수가 1로 줄어들면 `expand_dims`를
사용해 ndim이 최소 2는 되게 한다.

__인자__

- __x__: `ndim >= 2`인 케라스 텐서 또는 변수.
- __y__: `ndim >= 2`인 케라스 텐서 또는 변수.
- __axes__: 대상 차원을 나타내는 int 리스트 (또는 int 한 개).
    `axes[0]`과 `axes[1]`의 길이가 같아야 한다.

__반환__

(합을 수행하는 차원을 뺀) `x`의 형태와 (배치 차원 및
합을 수행하는 차원을 뺀) `y`의 형태를 이어 붙인 것과
형태가 같은 텐서.
최종 랭크가 1이면 `(batch_size, 1)` 형태로 만든다.

__예시__

`x = [[1, 2], [3, 4]]`이고 `y = [[5, 6], [7, 8]]`이라고 할 때
`batch_dot(x, y, axes=1) = [[17], [53]]`이다.
이는 `x.dot(y.T)`의 주대각선이기도 한데 대각선 밖의 원소를
계산할 필요가 전혀 없다.

형태 추론:
`x`의 형태가 `(100, 20)`이고 `y`의 형태가 `(100, 30, 20)`이라고
하자. `axes`가 (1, 2)일 때 결과 텐서의 출력 형태를 알려면
`x`의 형태와 `y`의 형태의 각 차원을 돌면 된다.

* `x.shape[0]` : 100 : 출력 형태에 덧붙임.
* `x.shape[1]` : 20 : 출력 형태에 덧붙이지 않음.
`x`의 1번 차원 상에서 합을 수행. (`axes[0]` = 1)
* `y.shape[0]` : 100 : 출력 형태에 덧붙이지 않음.
`y`의 첫 번째 차원을 항상 무시.
* `y.shape[1]` : 30 : 출력 형태에 덧붙임.
* `y.shape[2]` : 20 : 출력 형태에 덧붙이지 않음.
`y`의 2번 차원 상에서 합을 수행. (`axes[1]` = 2)

`output_shape` = `(100, 30)`

```python
>>> x_batch = K.ones(shape=(32, 20, 1))
>>> y_batch = K.ones(shape=(32, 30, 20))
>>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
>>> K.int_shape(xy_batch_dot)
(32, 1, 30)
```

----

### transpose


```python
keras.backend.transpose(x)
```


텐서를 전치해서 반환한다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

__예시__

```python
>>> var = K.variable([[1, 2, 3], [4, 5, 6]])
>>> K.eval(var)
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]], dtype=float32)
>>> var_transposed = K.transpose(var)
>>> K.eval(var_transposed)
array([[ 1.,  4.],
       [ 2.,  5.],
       [ 3.,  6.]], dtype=float32)
```

```python
>>> inputs = K.placeholder((2, 3))
>>> inputs
<tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
>>> input_transposed = K.transpose(inputs)
>>> input_transposed
<tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

```

----

### gather


```python
keras.backend.gather(reference, indices)
```


텐서 `reference`에서 인덱스 `indices`의 원소들을 뽑아낸다.

__인자__

- __reference__: 텐서.
- __indices__: 인덱스들로 이뤄진 정수 텐서.

__반환__

`reference`와 타입이 같은 텐서.

----

### max


```python
keras.backend.max(x, axis=None, keepdims=False)
```


텐서 내 최댓값.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 최댓값을 찾을 축.
- __keepdims__: 불리언. 차원을 유지할 것인지 여부.
    `keepdims`가 `False`이면 텐서의 랭크가 1만큼
    줄어든다. `keepdims`가 `True`이면
    그 줄어든 차원을 길이 1짜리로 유지한다.

__반환__

`x`의 최댓값들로 된 텐서.

----

### min


```python
keras.backend.min(x, axis=None, keepdims=False)
```


텐서 내 최솟값.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 최솟값을 찾을 축.
- __keepdims__: 불리언. 차원을 유지할 것인지 여부.
    `keepdims`가 `False`이면 텐서의 랭크가 1만큼
    줄어든다. `keepdims`가 `True`이면
    그 줄어든 차원을 길이 1짜리로 유지한다.

__반환__

`x`의 최솟값들로 된 텐서.

----

### sum


```python
keras.backend.sum(x, axis=None, keepdims=False)
```


지정한 축을 따라 계산한 텐서 내 값들의 합.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 합을 계산할 축.
- __keepdims__: 불리언. 차원을 유지할 것인지 여부.
    `keepdims`가 `False`이면 텐서의 랭크가 1만큼
    줄어든다. `keepdims`가 `True`이면
    그 줄어든 차원을 길이 1짜리로 유지한다.

__반환__

`x`의 합으로 된 텐서.

----

### prod


```python
keras.backend.prod(x, axis=None, keepdims=False)
```


지정한 축을 따라 계산한 텐서 내 값들의 곱.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 곱을 계산할 축.
- __keepdims__: 불리언. 차원을 유지할 것인지 여부.
    `keepdims`가 `False`이면 텐서의 랭크가 1만큼
    줄어든다. `keepdims`가 `True`이면
    그 줄어든 차원을 길이 1짜리로 유지한다.

__반환__

`x`의 원소들의 곱으로 된 텐서.

----

### cumsum


```python
keras.backend.cumsum(x, axis=0)
```


지정한 축을 따라 계산한 텐서 내 값들의 누계 합(cumulative sum).

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 합을 계산할 축.

__반환__

`axis` 상으로 `x`의 값들의 누계 합으로 된 텐서.

----

### cumprod


```python
keras.backend.cumprod(x, axis=0)
```


지정한 축을 따라 계산한 텐서 내 값들의 누계 곱(cumulative product).

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 곱을 계산할 축.

__반환__

`axis` 상으로 `x`의 값들의 누계 곱으로 된 텐서.

----

### var


```python
keras.backend.var(x, axis=None, keepdims=False)
```


지정한 축을 따라 계산한 텐서의 분산.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 분산을 계산할 축.
- __keepdims__: 불리언. 차원을 유지할 것인지 여부.
    `keepdims`가 `False`이면 텐서의 랭크가 1만큼
    줄어든다. `keepdims`가 `True`이면
    그 줄어든 차원을 길이 1짜리로 유지한다.

__반환__

`x`의 원소들의 분산으로 된 텐서.

----

### std


```python
keras.backend.std(x, axis=None, keepdims=False)
```


지정한 축을 따라 계산한 텐서의 표준 편자.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 표준 편차를 계산할 축.
- __keepdims__: 불리언. 차원을 유지할 것인지 여부.
    `keepdims`가 `False`이면 텐서의 랭크가 1만큼
    줄어든다. `keepdims`가 `True`이면
    그 줄어든 차원을 길이 1짜리로 유지한다.

__반환__

`x`의 원소들의 표준 편차로 된 텐서.

----

### mean


```python
keras.backend.mean(x, axis=None, keepdims=False)
```


지정한 축을 따라 계산한 텐서의 평균.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 평균을 계산할 축.
- __keepdims__: 불리언. 차원을 유지할 것인지 여부.
    `keepdims`가 `False`이면 텐서의 랭크가 1만큼
    줄어든다. `keepdims`가 `True`이면
    그 줄어든 차원을 길이 1짜리로 유지한다.

__반환__

`x`의 원소들의 평균로 된 텐서.

----

### any


```python
keras.backend.any(x, axis=None, keepdims=False)
```


비트로 축소 (논리 OR).

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 축소를 수행할 축.
- __keepdims__: 축소 축을 버릴 것인지 브로드캐스트 할 것인지.

__반환__

(0과 1로 된) uint8 텐서.

----

### all


```python
keras.backend.all(x, axis=None, keepdims=False)
```


비트로 축소 (논리 AND).

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 축소를 수행할 축.
- __keepdims__: 축소 축을 버릴 것인지 브로드캐스트 할 것인지.

__반환__

(0과 1로 된) uint8 텐서.

----

### argmax


```python
keras.backend.argmax(x, axis=-1)
```


축 상의 최댓값의 인덱스를 반환한다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 축소를 수행할 축.

__반환__

텐서.

----

### argmin


```python
keras.backend.argmin(x, axis=-1)
```


축 상의 최솟값의 인덱스를 반환한다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 축소를 수행할 축.

__반환__

텐서.

----

### square


```python
keras.backend.square(x)
```


원소별 제곱.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### abs


```python
keras.backend.abs(x)
```


원소별 절대값.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### sqrt


```python
keras.backend.sqrt(x)
```


원소별 제곱근.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### exp


```python
keras.backend.exp(x)
```


원소별 지수.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### log


```python
keras.backend.log(x)
```


원소별 로그.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### logsumexp


```python
keras.backend.logsumexp(x, axis=None, keepdims=False)
```


log(sum(exp(텐서의 여러 차원에 걸친 원소들)))을 계산한다.

이 함수는 log(sum(exp(x)))보다 수치적으로 더 안정적이다.
큰 입력에 exp를 해서 발생하는 오버플로우와 작은 입력에
log를 해서 발생하는 언더플로우를 피할 수 있다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 정수. 축소를 할 축.
- __keepdims__: 불리언. 차원을 유지할 것인지 여부.
    `keepdims`가 `False`이면 텐서의 랭크가 1만큼
    줄어든다. `keepdims`가 `True`이면
    그 줄어든 차원을 길이 1짜리로 유지한다.

__반환__

축소된 텐서.

----

### round


```python
keras.backend.round(x)
```


원소별 반올림.

중간값인 경우 "짝수 쪽으로" 방식 사용.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### sign


```python
keras.backend.sign(x)
```


원소별 부호.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### pow


```python
keras.backend.pow(x, a)
```


원소별 거듭제곱.

__인자__

- __x__: 텐서 또는 변수.
- __a__: 파이썬 정수.

__반환__

텐서.

----

### clip


```python
keras.backend.clip(x, min_value, max_value)
```


원소별 값 범위 자르기.

__인자__

- __x__: 텐서 또는 변수.
- __min_value__: 파이썬 실수 또는 정수.
- __max_value__: 파이썬 실수 또는 정수.

__반환__

텐서.

----

### equal


```python
keras.backend.equal(x, y)
```


두 텐서 간의 원소별 동일 여부.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

bool 텐서.

----

### not_equal


```python
keras.backend.not_equal(x, y)
```


두 텐서 간의 원소별 비동일 여부.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

bool 텐서.

----

### greater


```python
keras.backend.greater(x, y)
```


원소별 (x > y)의 진리치.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

bool 텐서.

----

### greater_equal


```python
keras.backend.greater_equal(x, y)
```


원소별 (x >= y)의 진리치.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

bool 텐서.

----

### less


```python
keras.backend.less(x, y)
```


원소별 (x < y)의 진리치.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

bool 텐서.

----

### less_equal


```python
keras.backend.less_equal(x, y)
```


원소별 (x <= y)의 진리치.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

bool 텐서.

----

### maximum


```python
keras.backend.maximum(x, y)
```


두 텐서의 원소별 최댓값.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

텐서.

----

### minimum


```python
keras.backend.minimum(x, y)
```


두 텐서의 원소별 최솟값.

__인자__

- __x__: 텐서 또는 변수.
- __y__: 텐서 또는 변수.

__반환__

텐서.

----

### sin


```python
keras.backend.sin(x)
```


x의 원소별 sin을 계산한다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### cos


```python
keras.backend.cos(x)
```


x의 원소별 cos을 계산한다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### normalize_batch_in_training


```python
keras.backend.normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.001)
```


배치에 대해 평균과 표준 편차를 계산한 다음 배치에 batch_normalization을 적용한다.

__인자__

- __x__: 입력 텐서 또는 변수.
- __gamma__: 입력 크기 조정에 쓸 텐서.
- __beta__: 입력을 중심으로 조정할 텐서.
- __reduction_axes__: 정수들의 이터러블.
    정규화할 축들.
- __epsilon__: 퍼징 인자.

__반환__

길이 3짜리 튜플 `(정규화된_텐서, 평균, 분산)`.

----

### batch_normalization


```python
keras.backend.batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=0.001)
```


주어진 mean, var, beta, gamma로 x에 배치 정규화를 적용한다.

즉 이걸 반환한다:
`output = (x - mean) / sqrt(var + epsilon) * gamma + beta`

__인자__

- __x__: 입력 텐서 또는 변수.
- __mean__: 배치의 평균.
- __var__: 배치의 분산.
- __beta__: 입력을 중심으로 조정할 텐서.
- __gamma__: 입력 크기 조정에 쓸 텐서.
- __axis__: 정수. 정규화를 해야 할 축
    (보통은 피쳐 축).
- __epsilon__: 퍼징 인자.

__반환__

텐서.

----

### concatenate


```python
keras.backend.concatenate(tensors, axis=-1)
```


지정한 축을 따라 텐서들의 목록을 접합한다.

__인자__

- __tensors__: 접합할 텐서들의 목록.
- __axis__: 접합 축.

__반환__

텐서.

----

### reshape


```python
keras.backend.reshape(x, shape)
```


지정한 형태로 텐서의 형태를 바꾼다.

__인자__

- __x__: 텐서 또는 변수.
- __shape__: 목적 형태 튜플.

__반환__

텐서.

----

### permute_dimensions


```python
keras.backend.permute_dimensions(x, pattern)
```


텐서의 축들을 치환한다.

__인자__

- __x__: 텐서 또는 변수.
- __pattern__: 차원 인덱스들의 튜플.
    가령 `(0, 2, 1)`.

__반환__

텐서.

----

### resize_images


```python
keras.backend.resize_images(x, height_factor, width_factor, data_format)
```


4차원 텐서에 들어 있는 이미지들의 크기를 바꾼다.

__인자__

- __x__: 크기를 조정할 텐서 또는 변수.
- __height_factor__: 양의 정수.
- __width_factor__: 양의 정수.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.

__반환__

텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나 `"channels_first"`가 아님.

----

### resize_volumes


```python
keras.backend.resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
```


5차원 텐서에 들어 있는 볼륨들의 크기를 바꾼다.

__인자__

- __x__: 크기를 조정할 텐서 또는 변수.
- __depth_factor__: 양의 정수.
- __height_factor__: 양의 정수.
- __width_factor__: 양의 정수.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.

__반환__

텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`이나 `"channels_first"`가 아님.

----

### repeat_elements


```python
keras.backend.repeat_elements(x, rep, axis)
```


`np.repeat`처럼 축을 따라 텐서의 원소들을 반복한다.

`x`의 형태가 `(s1, s2, s3)`이고 `axis`가 `1`이면
출력이 `(s1, s2 * rep, s3)` 형태가 된다.

__인자__

- __x__: 텐서 또는 변수.
- __rep__: 파이썬 정수. 반복 횟수.
- __axis__: 따라 반복을 할 축.

__반환__

텐서.

----

### repeat


```python
keras.backend.repeat(x, n)
```


2차원 텐서를 반복한다.

`x`의 형태가 (samples, dim)이고 `n`이 `2`이면
출력이 `(samples, 2, dim)` 형태가 된다.

__인자__

- __x__: 텐서 또는 변수.
- __n__: 파이썬 정수. 반복 횟수.

__반환__

텐서.

----

### arange


```python
keras.backend.arange(start, stop=None, step=1, dtype='int32')
```


정수 열을 담은 1차원 텐서를 만든다.

함수 인자에서 테아노의 arange와 같은 관례를 따른다.
즉 인자를 하나만 주면 그게 실제로는 "stop" 인자이고
"start"가 0이다.

반환되는 텐서의 기본 타입은 텐서플로우의 기본 타입과
일치하는 `'int32'`이다.

__인자__

- __start__: 시작하는 값.
- __stop__: 멈추는 값.
- __step__: 연속한 두 값의 차이.
- __dtype__: 사용할 정수 dtype.

__반환__

정수 텐서.


----

### tile


```python
keras.backend.tile(x, n)
```


`x`를 `n`만큼 타일처럼 반복한 텐서를 만든다.

__인자__

- __x__: 텐서 또는 변수.
- __n__: 정수 리스트. 길이가 `x`의 차원 개수와 같아야 한다.

__반환__

타일처럼 반복한 텐서.

----

### flatten


```python
keras.backend.flatten(x)
```


텐서를 편평하게 만든다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

1차원으로 형태가 바뀐 텐서.

----

### batch_flatten


```python
keras.backend.batch_flatten(x)
```


n차원 텐서를 0번째 차원이 동일한 2차원 텐서로 바꾼다.

달리 말해 배치의 각 데이터 표본을 편평하게 만든다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### expand_dims


```python
keras.backend.expand_dims(x, axis=-1)
```


인덱스 "axis"에 크기 1인 차원을 추가한다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 새 축을 추가할 위치.

__반환__

차원이 확장된 텐서.

----

### squeeze


```python
keras.backend.squeeze(x, axis)
```


인덱스 "axis"에서 텐서의 1짜리 차원을 제거한다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 버릴 축.

__반환__

`x`와 데이터가 같되 차원이 축소된 텐서.

----

### temporal_padding


```python
keras.backend.temporal_padding(x, padding=(1, 1))
```


3차원 텐서의 가운데 차원에 패딩을 덧붙인다.

__인자__

- __x__: 텐서 또는 변수.
- __padding__: 정수 2개짜리 튜플. 1번 차원 시작과 끝에
    0을 몇 개씩 추가할 것인지.

__반환__

패딩을 덧붙인 3차원 텐서.

----

### spatial_2d_padding


```python
keras.backend.spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None)
```


4차원 텐서의 2번째와 3번째 차원에 패딩을 덧붙인다.

__인자__

- __x__: 텐서 또는 변수.
- __padding__: 튜플 2개짜리 튜플. 패딩 패턴.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.

__반환__

패딩을 덧붙인 4차원 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`이나 `"channels_first"`가 아님.

----

### spatial_3d_padding


```python
keras.backend.spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None)
```


5차원 텐서의 깊이, 높이, 너비 차원을 따라 0을 덧붙인다.

그 차원들 각각의 왼쪽과 오른쪽에
"padding[0]"개, "padding[1]"개, "padding[2]"개씩 0을 덧붙인다.

data_format이 'channels_last'면
2번째, 3번째, 4번째 차원에 패딩을 덧붙인다.
data_format이 'channels_first'면
3번째, 4번째, 5번째 차원에 패딩을 덧붙인다.

__인자__

- __x__: 텐서 또는 변수.
- __padding__: 튜플 3개짜리 튜플. 패딩 패턴.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.

__반환__

패딩을 덧붙인 5차원 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`이나 `"channels_first"`가 아님.


----

### stack


```python
keras.backend.stack(x, axis=0)
```


랭크 `R`인 텐서들을 쌓아서 랭크 `R+1`인 텐서로 만든다.

__인자__

- __x__: 텐서들의 리스트.
- __axis__: 따라서 쌓기를 수행할 축.

__반환__

텐서.

----

### one_hot


```python
keras.backend.one_hot(indices, num_classes)
```


정수 텐서의 원핫 표현을 계산한다.

__인자__

- __indices__: `(batch_size, dim1, dim2, ... dim(n-1))`
    형태의 n차원 정수 텐서.
- __num_classes__: 정수. 고려할 클래스 수.

__반환__

`(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
형태인 입력의 (n + 1)차원 원핫 표현.

----

### reverse


```python
keras.backend.reverse(x, axes)
```


지정한 축들을 따라 텐서를 뒤집는다.

__인자__

- __x__: 뒤집을 텐서.
- __axes__: 정수 또는 정수들의 이터러블.
    뒤집을 축들.

__반환__

텐서.

----

### slice


```python
keras.backend.slice(x, start, size)
```


텐서에서 조각을 추출한다.

__인자__

- __x__: 입력 텐서.
- __start__: 정수 리스트/튜플 또는 텐서.
    각 축에서 조각의 시작 인덱스를 나타냄.
- __size__: 정수 리스트/튜플 또는 텐서.
    각 축에서 조각의 차원 개수를 나타냄.

__반환__

텐서
```
x[start[0]: start[0] + size[0],
  ...,
  start[-1]: start[-1] + size[-1]]
```

----

### get_value


```python
keras.backend.get_value(x)
```


변수의 값을 반환한다.

__인자__

- __x__: 입력 변수.

__반환__

Numpy 배열.

----

### batch_get_value


```python
keras.backend.batch_get_value(ops)
```


여러 텐서 변수의 값을 반환한다.

__인자__

- __ops__: 실행할 연산들의 리스트.

__반환__

Numpy 배열들의 리스트.

----

### set_value


```python
keras.backend.set_value(x, value)
```


Numpy 배열을 가지고 변수의 값을 설정한다.

__인자__

- __x__: 새 값을 설정할 텐서.
- __value__: 텐서에 설정할 값. Numpy 배열 (동일 형태).

----

### batch_set_value


```python
keras.backend.batch_set_value(tuples)
```


여러 텐서 변수들의 값을 한 번에 설정한다.

__인자__

- __tuples__: `(텐서, 값)` 튜플들의 리스트.
    `value`가 Numpy 배열이어야 함.

----

### print_tensor


```python
keras.backend.print_tensor(x, message='')
```


`message`와 텐서 평가 값을 찍는다.

참고로 `print_tensor`는 `x`와 동일한 새 텐서를 반환하는데
이어지는 코드에서 그 텐서를 써야 한다. 안 그러면
평가 때 print 동작이 계산에 들어가지 않게 된다.

__예시__

```python
>>> x = K.print_tensor(x, message="x is: ")
```

__인자__

- __x__: 찍을 텐서.
- __message__: 텐서와 함께 찍을 메시지.

__반환__

변경 없는 동일한 텐서 `x`.

----

### function


```python
keras.backend.function(inputs, outputs, updates=None)
```


케라스 함수를 만든다.

__인자__

- __inputs__: 플레이스홀더 텐서들의 리스트.
- __outputs__: 출력 텐서들의 리스트.
- __updates__: 갱신 연산들의 리스트.
- __**kwargs__: `tf.Session.run`으로 전달.

__반환__

Numpy 배열인 출력 값들.

__예외__

- __ValueError__: 유효하지 않은 kwargs를 준 경우.

----

### gradients


```python
keras.backend.gradients(loss, variables)
```


`variables`에 대한 `loss`의 경사를 반환한다.

__인자__

- __loss__: 최소화 할 스칼라 텐서.
- __variables__: 변수들의 리스트.

__반환__

경사 텐서.

----

### stop_gradient


```python
keras.backend.stop_gradient(variables)
```


다른 모든 변수에 대해 경사가 0인 `variables`를 반환한다.

__인자__

- __variables__: 다른 모든 변수에 대해 상수로 여기게 할
    텐서 또는 텐서들의 리스트.

__반환__

다른 모든 변수에 대해 상수 경사인 (받은 인자에 따라서)
    단일 텐서 또는 텐서들의 리스트

----

### rnn


```python
keras.backend.rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


텐서의 시간 차원 상에서 반복 동작을 한다.

__인자__

- __step_function__:
    - 매개변수:
        - inputs: (samples, ...) 형태의 텐서 (시간 차원 없음).
            특정 time step에서 표본 배치를 위한 입력을 나타냄.
        - states: 텐서들의 리스트.
    - 반환:
        - outputs: (samples, ...) 형태의 텐서. (시간 차원 없음)
        - new_states: 텐서들의 리스트. 'states'와 길이 및 형태가
            동일.
- __inputs__: (samples, time, ...) 형태의 시간 데이터 텐서
    (최소 3차원).
- __initial_states__: (samples, ...) 형태의 텐서 (시간 차원 없음).
    단계 함수에서 쓰는 상태의 초기값을 담음.
- __go_backwards__: 불리언. True이면 시간 차원을 역순으로
    반복하고서 뒤집힌 열을 반환한다.
- __mask__: (samples, time) 형태의 이진 텐서.
    마스킹 하는 원소에는 0.
- __constants__: 각 단계로 전달할 상수 값들의 리스트.
- __unroll__: RNN을 전개할지 아니면 심볼릭 루프를
    (백엔드에 따라 `while_loop`이나 `scan`을) 사용할지.
- __input_length__: 입력의 timestep 수.

__반환__

튜플 `(last_output, outputs, new_states)`.

- last_output: rnn의 마지막 출력. `(samples, ...)` 형태.
- outputs: `(samples, time, ...)` 형태의 텐서.
`outputs[s, t]`가 표본 `s`에 대한 시간 `t`에서의
단계 함수 출력.
- new_states: 텐서들의 리스트, 단계 함수가 반환한
마지막 상태. `(samples, ...)` 형태.

__예외__

- __ValueError__: 입력 차원이 3개가 안 되는 경우.
- __ValueError__: `unroll`이 `True`인데 입력 timestep이
    고정된 수가 아닌 경우.
- __ValueError__: `mask`를 줬는데 (not `None`)
    상태는 주지 않은 (`len(states)` == 0) 경우.

----

### switch


```python
keras.backend.switch(condition, then_expression, else_expression)
```


스칼라 값에 따라 두 연산 중 하나로 진행한다.

`then_expression`과 `else_espression` 모두 심볼릭 텐서여야
하며 *같은 형태여야* 한다.

__인자__

- __condition__: 텐서 (`int` 또는 `bool`).
- __then_expression__: 텐서 또는 텐서를 반환하는 호출 가능 객체.
- __else_expression__: 텐서 또는 텐서를 반환하는 호출 가능 객체.

__반환__

선택된 텐서.

__예외__

- __ValueError__: `condition`의 랭크가 식의 랭크보다 큰 경우.

----

### in_train_phase


```python
keras.backend.in_train_phase(x, alt, training=None)
```


훈련 단계면 `x`를 선택하고 아니면 `alt`를 선택한다.

`alt`가 `x`와 *같은 형태여야* 한다.

__인자__

- __x__: 훈련 단계에서 반환할 무언가.
    (텐서 또는 텐서를 반환하는 호출 가능 객체.)
- __alt__: 아니면 반환할 무언가.
    (텐서 또는 텐서를 반환하는 호출 가능 객체.)
- __training__: 선택적. 학습 단계를 나타내는
    스칼라 텐서
    (파이썬 불리언 또는 파이썬 정수).

__반환__

`training` 플래그에 따라 `x` 아니면 `alt`.
`training` 플래그 기본값은 `K.learning_phase()`이다.

----

### in_test_phase


```python
keras.backend.in_test_phase(x, alt, training=None)
```


테스트 단계면 `x`를 선택하고 아니면 `alt`를 선택한다.

`alt`가 `x`와 *같은 형태여야* 한다.

__인자__

- __x__: 테스트 단계에서 반환할 무언가.
    (텐서 또는 텐서를 반환하는 호출 가능 객체.)
- __alt__: 아니면 반환할 무언가.
    (텐서 또는 텐서를 반환하는 호출 가능 객체.)
- __training__: 선택적. 학습 단계를 나타내는
    스칼라 텐서
    (파이썬 불리언 또는 파이썬 정수).

__반환__

`K.learing_phase()`에 따라 `x` 아니면 `alt`.

----

### relu


```python
keras.backend.relu(x, alpha=0.0, max_value=None)
```


정류 선형 단위.

기본값으로는 원소별 `max(x, 0)`을 반환한다.

__인자__

- __x__: 텐서 또는 변수.
- __alpha__: 스칼라. 음수 구역의 기울기 (기본값=`0.`).
- __max_value__: 포화 한계치.

__반환__

텐서.

----

### elu


```python
keras.backend.elu(x, alpha=1.0)
```


지수 선형 단위.

__인자__

- __x__: 활성 함수를 계산할 텐서 또는 변수.
- __alpha__: 스칼라. 음수 구역의 기울기.

__반환__

텐서.

----

### softmax


```python
keras.backend.softmax(x, axis=-1)
```


텐서의 소프트맥스.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 소프트맥스를 수행할 차원.
    기본값은 -1이고 마지막 차원을 나타낸다.

__반환__

텐서.

----

### softplus


```python
keras.backend.softplus(x)
```


텐서의 softplus.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### softsign


```python
keras.backend.softsign(x)
```


텐서의 softsign.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### categorical_crossentropy


```python
keras.backend.categorical_crossentropy(target, output, from_logits=False, axis=-1)
```


출력 텐서와 목표 텐서 간의 범주 크로스 엔트로피.

__인자__

- __target__: `output`과 형태가 같은 텐서.
- __output__: 소프트맥스 결과 텐서.
    (단 `from_logits`이 True이면
    `output`이 로짓이라고 기대함.)
- __from_logits__: 불리언. `output`이 소프트맥스
    결과인지 아니면 로짓 텐서인지.
- __axis__: 채널 축을 지정하는 정수. `axis=-1`은
    데이터 형식 `channels_last`에 해당하고
    `axis=1`은 데이터 형식 `channels_first`에
    해당함.

__반환__

출력 텐서.

__예외__

- __ValueError__: `axis`가 -1이나 `output`의 축들 중
    하나가 아닌 경우.

----

### sparse_categorical_crossentropy


```python
keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1)
```


정수 목표의 범주 크로스 엔트로피.

__인자__

- __target__: 정수 텐서.
- __output__: 소프트맥스 결과 텐서.
    (단 `from_logits`이 True이면
    `output`이 로짓이라고 기대함.)
- __from_logits__: 불리언. `output`이 소프트맥스
    결과인지 아니면 로짓 텐서인지.
- __axis__: 채널 축을 지정하는 정수. `axis=-1`은
    데이터 형식 `channels_last`에 해당하고
    `axis=1`은 데이터 형식 `channels_first`에
    해당함.

__반환__

출력 텐서.

__예외__

- __ValueError__: `axis`가 -1이나 `output`의 축들 중
    하나가 아닌 경우.

----

### binary_crossentropy


```python
keras.backend.binary_crossentropy(target, output, from_logits=False)
```


출력 텐서와 목표 텐서 간의 이진 크로스 엔트로피.

__인자__

- __target__: `output`과 형태가 같은 텐서.
- __output__: 텐서.
- __from_logits__: `output`이 로짓 텐서라고 기대할지 여부.
    기본적으로 `output`이 확률 분포를 담고 있다고 본다.

__반환__

텐서.

----

### sigmoid


```python
keras.backend.sigmoid(x)
```


원소별 시그모이드.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### hard_sigmoid


```python
keras.backend.hard_sigmoid(x)
```


구간별 시그모이드 선형 근사.

시그모이드보다 빠르다.
`x < -2.5`이면 `0.`, `x > 2.5`이면 `1.`을 반환한다.
`-2.5 <= x <= 2.5`이면 `0.2 * x + 0.5`를 반환한다.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### tanh


```python
keras.backend.tanh(x)
```


원소별 쌍곡탄젠트.

__인자__

- __x__: 텐서 또는 변수.

__반환__

텐서.

----

### dropout


```python
keras.backend.dropout(x, level, noise_shape=None, seed=None)
```


`x`의 항들을 무작위로 0으로 설정하면서 그에 맞게 텐서 전체의 값을 조정한다.

__인자__

- __x__: 텐서.
- __level__: 텐서의 항들에서 0으로 설정할 비율.
- __noise_shape__: 난수적으로 생성되는 유지/탈락 플래그들의 형태.
    `x`의 형태로 브로드캐스트 가능해야 함.
- __seed__: 결정론적 동작 보장을 위한 난수 시드.

__반환__

텐서.

----

### l2_normalize


```python
keras.backend.l2_normalize(x, axis=None)
```


지정한 축을 따라서 L2 norm에 따라 텐서를 정규화 한다.

__인자__

- __x__: 텐서 또는 변수.
- __axis__: 따라서 정규화를 수행할 축.

__반환__

텐서.

----

### in_top_k


```python
keras.backend.in_top_k(predictions, targets, k)
```


`targets`가 `predictions`의 상위 `k`개에 들어가는지 여부를 반환한다.

__인자__

- __predictions__: 형태가 `(batch_size, classes)`이고 타입이 `float32`인 텐서.
- __targets__: 길이가 `batch_size`이고 타입이 `int32`나 `int64`인 1차원 텐서.
- __k__: `int`. 고려할 상위 항목 개수.

__반환__

길이가 `batch_size`이고 타입이 `bool`인 1차원 텐서.
`predictions[i, targets[i]]`가 `predictions[i]`의 상위 `k`개 값에
들어가면 `output[i]`가 `True`이다.

----

### conv1d


```python
keras.backend.conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


1차원 합성곱.

__인자__

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서.
- __strides__: 보폭 정수.
- __padding__: 문자열. `"same"` 또는 `"causal"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
- __dilation_rate__: 정수 팽창률.

__반환__

1차원 합성곱의 결과인 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### conv2d


```python
keras.backend.conv2d(x, kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2차원 합성곱.

__인자__

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서.
- __strides__: 보폭 튜플.
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
    입력/커널/출력에 테아노와 텐서플로우/CNTK의 데이터 형식 중
    어느 쪽을 쓸 것인지.
- __dilation_rate__: 정수 2개짜리 튜플.

__반환__

2차원 합성곱의 결과인 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### conv2d_transpose


```python
keras.backend.conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid', data_format=None)
```


2차원 역합성곱 (전치 합성곱).

__인자__

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서.
- __output_shape__: 출력 형태를 나타내는 1차원 int 텐서.
- __strides__: 보폭 튜플.
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
    입력/커널/출력에 테아노와 텐서플로우/CNTK의 데이터 형식 중
    어느 쪽을 쓸 것인지.

__반환__

전치 2차원 합성곱의 결과인 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### separable_conv1d


```python
keras.backend.separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


분리식 필터의 1차원 합성곱.

__인자__

- __x__: 입력 텐서.
- __depthwise_kernel__: 깊이 방향 합성곱을 위한 합성곱 커널.
- __pointwise_kernel__: 1x1 합성곱을 위한 커널.
- __strides__: 보폭 정수.
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
- __dilation_rate__: 정수 팽창률.

__반환__

출력 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### separable_conv2d


```python
keras.backend.separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


분리식 필터의 2차원 합성곱.

__인자__

- __x__: 입력 텐서.
- __depthwise_kernel__: 깊이 방향 합성곱을 위한 합성곱 커널.
- __pointwise_kernel__: 1x1 합성곱을 위한 커널.
- __strides__: 보폭 튜플 (길이 2).
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
- __dilation_rate__: 정수들의 튜플.
    분리식 합성곱의 팽창률.

__반환__

출력 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### depthwise_conv2d


```python
keras.backend.depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


분리식 필터의 2차원 합성곱.

__인자__

- __x__: 입력 텐서.
- __depthwise_kernel__: 깊이 방향 합성곱을 위한 합성곱 커널.
- __strides__: 보폭 튜플 (길이 2).
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
- __dilation_rate__: 정수들의 튜플.
    분리식 합성곱의 팽창률.

__반환__

출력 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### conv3d


```python
keras.backend.conv3d(x, kernel, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1))
```


3차원 합성곱.

__인자__

- __x__: 텐서 또는 변수.
- __kernel__: 커널 텐서.
- __strides__: 보폭 튜플.
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
    입력/커널/출력에 테아노와 텐서플로우/CNTK의 데이터 형식 중
    어느 쪽을 쓸 것인지.
- __dilation_rate__: 정수 3개짜리 튜플.

__반환__

3차원 합성곱의 결과인 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### conv3d_transpose


```python
keras.backend.conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1), padding='valid', data_format=None)
```


3차원 역합성곱 (전치 합성곱).

__인자__

- __x__: 입력 텐서.
- __kernel__: 커널 텐서.
- __output_shape__: 출력 형태를 나타내는 1차원 int 텐서.
- __strides__: 보폭 튜플.
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
    입력/커널/출력에 테아노와 텐서플로우/CNTK의 데이터 형식 중
    어느 쪽을 쓸 것인지.

__반환__

전치 3차원 합성곱의 결과인 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### pool2d


```python
keras.backend.pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max')
```


2차원 풀링.

__인자__

- __x__: 텐서 또는 변수.
- __pool_size__: 정수 2개짜리 튜플.
- __strides__: 정수 2개짜리 튜플.
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
- __pool_mode__: 문자열. `"max"` 또는 `"avg"`.

__반환__

2차원 풀링의 결과인 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나 `"channels_first"`가 아님.
- __ValueError__: `pool_mode`가 `"max"`나 `"avg"`가 아님.

----

### pool3d


```python
keras.backend.pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max')
```


3차원 풀링.

__인자__

- __x__: 텐서 또는 변수.
- __pool_size__: 정수 3개짜리 튜플.
- __strides__: 정수 3개짜리 튜플.
- __padding__: 문자열. `"same"` 또는 `"valid"`.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.
- __pool_mode__: 문자열. `"max"` 또는 `"avg"`.

__반환__

3차원 풀링의 결과인 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나 `"channels_first"`가 아님.
- __ValueError__: `pool_mode`가 `"max"`나 `"avg"`가 아님.


----

### bias_add


```python
keras.backend.bias_add(x, bias, data_format=None)
```


텐서에 편향 벡터를 더한다.

__인자__

- __x__: 텐서 또는 변수.
- __bias__: 더할 편향 벡터.
- __data_format__: 문자열. `"channels_last"` 또는 `"channels_first"`.

__반환__

출력 텐서.

__예외__

- __ValueError__: 다음 두 경우 중 하나일 때.
    1. `data_format` 인자가 유효하지 않음.
    2. 편향 형태가 유효하지 않음.
       편향은 ndim(x) - 1 차원의
       벡터 또는 텐서여야 함.

----

### random_normal


```python
keras.backend.random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


정규 분포인 값들의 텐서를 반환한다.

__인자__

- __shape__: 정수들의 튜플. 생성할 텐서의 형태.
- __mean__: float. 표본을 뽑을 정규 분포의 평균.
- __stddev__: float. 표본을 뽑을 정규 분포의
    표준 편차.
- __dtype__: 문자열. 반환 텐서의 dtype.
- __seed__: 정수. 난수 시드.

__반환__

텐서.

----

### random_uniform


```python
keras.backend.random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```


균일 분포인 값들의 텐서를 반환한다.

__인자__

- __shape__: 정수들의 튜플. 생성할 텐서의 형태.
- __minval__: float. 표본을 뽑을 균일 분포의 하한.
- __minval__: float. 표본을 뽑을 균일 분포의 상한.
- __dtype__: 문자열. 반환 텐서의 dtype.
- __seed__: 정수. 난수 시드.

__반환__

텐서.

----

### random_binomial


```python
keras.backend.random_binomial(shape, p=0.0, dtype=None, seed=None)
```


난수 이항 분포인 값들의 텐서를 반환한다.

__인자__

- __shape__: 정수들의 튜플. 생성할 텐서의 형태.
- __p__: float. `0. <= p <= 1`. 이항 분포 확률.
- __dtype__: 문자열. 반환 텐서의 dtype.
- __seed__: 정수. 난수 시드.

__반환__

텐서.

----

### truncated_normal


```python
keras.backend.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


절단 난수 정규 분포인 값들의 텐서를 반환한다.

생성되는 값들이 지정한 평균과 표준 편차의
정규 분포를 따르되 평균으로부터 표준 편차
두 배 넘게 떨어진 값은 버리고 다시 고른다.

__인자__

- __shape__: 정수들의 튜플. 생성할 텐서의 형태.
- __mean__: 값들의 평균.
- __stddev__: 값들의 표준 편차.
- __dtype__: 문자열. 반환 텐서의 dtype.
- __seed__: 정수. 난수 시드.

__반환__

텐서.

----

### ctc_label_dense_to_sparse


```python
keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
```


CTC 레이블을 밀집에서 희소로 변환한다.

__인자__

- __labels__: 밀집 CTC 레이블.
- __label_lengths__: 레이블들의 길이.

__반환__

레이블들의 희소 텐서 표현.

----

### ctc_batch_cost


```python
keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
```


각 배치 요소에 CTC 손실 알고리듬을 돌린다.

__인자__

- __y_true__: 진리치 레이블을 담은 텐서
    `(samples, max_string_length)`.
- __y_pred__: 예측 내지 소프트맥스 출력을 담은
    텐서 `(samples, time_steps, num_categories)`.
- __input_length__: `y_pred`의 각 배치 항목에 대한
    열 길이를 담은 텐서 `(samples, 1)`.
- __label_length__: `y_true`의 각 배치 항목에 대한
    열 길이를 담은 텐서 `(samples, 1)`.

__반환__

각 원소의 CTC 손실을 담은 (samples, 1) 형태의 텐서.

----

### ctc_decode


```python
keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
```


소프트맥스의 출력을 디코드 한다.

탐욕 탐색(최선 경로)을 쓸 수도 있고
제약된 사전 탐색을 쓸 수도 있다.

__인자__

- __y_pred__: 예측 내지 소프트맥스 출력을 담은
    텐서 `(samples, time_steps, num_categories)`.
- __input_length__: `y_pred`의 각 배치 항목에 대한
    열 길이를 담은 텐서 `(samples, )`.
- __greedy__: `true`이면 훨씬 빠른 최선 경로 탐색을 수행함.
    그 경우 사전을 쓰지 않음.
- __beam_width__: `greedy`가 `false`인 경우에, 이 빔 폭으로
    빔 탐색 디코더를 쓰게 된다.
- __top_paths__: `greedy`가 `false`인 경우에,
    가능성 높은 경로 몇 개를 반환할 것인지.

__반환__

- __튜플__:
    - 리스트: `greedy`가 `true`이면 디코딩 된 열을 담은
        윈소 하나짜리 리스트를 반환한다.
        `false`이면 가능성 높은 `top_paths` 개의
        디코딩 된 열을 반환한다.
        중요: 빈 레이블은 `-1`로 반환된다.
    - 디코딩 된 열 각각의 로그 확률을 담은
        텐서 `(top_paths, )`.

----

### map_fn


```python
keras.backend.map_fn(fn, elems, name=None, dtype=None)
```


원소 elems에 대해 함수 fn을 맵 하고 출력들을 반환한다.

__인자__

- __fn__: elems의 각 원소에 대해 호출할 호출 가능 객체.
- __elems__: 텐서.
- __name__: 그래프의 맵 노드의 문자열 이름.
- __dtype__: 출력 데이터 타입.

__반환__

dtype이 `dtype`인 텐서.

----

### foldl


```python
keras.backend.foldl(fn, elems, initializer=None, name=None)
```


fn으로 왼쪽에서 오른쪽으로 합치며 elems를 리듀스 한다.

__인자__

- __fn__: elems의 각 원소와 누산치에 대해 호출할 호출 가능 객체.
    예를 들면 `lambda acc, x: acc + x`.
- __elems__: 텐서.
- __initializer__: 처음 사용할 값 (None이면 `elems[0]`).
- __name__: 그래프의 foldl 노드의 문자열 이름.

__반환__

`initializer`와 타입 및 형태가 같은 텐서.

----

### foldr


```python
keras.backend.foldr(fn, elems, initializer=None, name=None)
```


fn으로 오른쪽에서 왼쪽으로 합치며 elems를 리듀스 한다.

__인자__

- __fn__: elems의 각 원소와 누산치에 대해 호출할 호출 가능 객체.
    예를 들면 `lambda acc, x: acc + x`.
- __elems__: 텐서.
- __initializer__: 처음 사용할 값 (None이면 `elems[-1]`).
- __name__: 그래프의 foldr 노드의 문자열 이름.

__반환__

`initializer`와 타입 및 형태가 같은 텐서.

----

### local_conv1d


```python
keras.backend.local_conv1d(inputs, kernel, kernel_size, strides, data_format=None)
```


비공유 가중치로 1차원 합성곱을 적용한다.

__인자__

- __inputs__: (batch_size, steps, input_dim) 형태의 3차원 텐서.
- __kernel__: 합성곱을 위한 비공유 가중치.
        (output_length, feature_dim, filters) 형태.
- __kernel_size__: 단일 정수 튜플.
        1차원 합성곱 윈도의 길이 지정.
- __strides__: 단일 정수 튜플. 합성곱 보폭 길이 지정.
- __data_format__: 데이터 형식. channels_first 또는 channels_last.

__반환__

비공유 가중치 1차원 합성곱 후의 텐서. (batch_size, output_length, filters) 형태.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### local_conv2d


```python
keras.backend.local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None)
```


비공유 가중치로 2차원 합성곱을 적용한다.

__인자__

- __inputs__: data_format='channels_first'이면
        (batch_size, filters, new_rows, new_cols)
        형태의 4차원 텐서.
        data_format='channels_last'이면
        (batch_size, new_rows, new_cols, filters)
        형태의 4차원 텐서.
- __kernel__: 합성곱을 위한 비공유 가중치.
        (output_items,, feature_dim, filters) 형태.
- __kernel_size__: 정수 2개짜리 튜플.
        2차원 합성곱 윈도의 너비와 높이 지정.
- __strides__: 정수 2개짜리 튜플.
         너비와 높이 방향의 합성곱 보폭 지정.
- __output_shape__: 튜플 (output_row, output_col).
- __data_format__: 데이터 형식. channels_first 또는 channels_last.

__반환__

data_format='channels_first'이면
(batch_size, filters, new_rows, new_cols)
형태의 4차원 텐서.
data_format='channels_last'이면
(batch_size, new_rows, new_cols, filters)
형태의 4차원 텐서.

__예외__

- __ValueError__: `data_format`이 `"channels_last"`나
    `"channels_first"`가 아닌 경우.

----

### backend


```python
keras.backend.backend()
```


현재 백엔드를 알아내기 위한
공개적으로 접근 가능한 메소드.

__반환__

문자열. 케라스에서 현재 쓰고 있는 백엔드의 이름.

__예시__

```python
>>> keras.backend.backend()
'tensorflow'
```

----

### get_uid


```python
keras.backend.get_uid(prefix='')
```


기본 그래프에 대한 uid를 얻는다.

__인자__

- __prefix__: 선택적. 그래프의 접두부.

__반환__

그 그래프에 대한 고유 식별자.

----

### reset_uids


```python
keras.backend.reset_uids()
```


그래프 식별자를 초기화 한다.

----

### clear_session


```python
keras.backend.clear_session()
```


현재 TF 그래프를 파기하고 새 그래프를 만든다.

이전 모델/층의 잡동사니들을 치우는 데 유용하다.

----

### manual_variable_initialization


```python
keras.backend.manual_variable_initialization(value)
```


수동 변수 초기화 플래그를 설정한다.

이 불리언 플래그는 인스턴스 생성 때 변수가
초기화 돼야 하는지 (기본), 아니면 사용자가
(가령 `tf.initialize_all_variables()`를 통해)
초기화를 맡아야 하는지 결정한다.

__인자__

- __value__: 파이썬 불리언.

----

### learning_phase


```python
keras.backend.learning_phase()
```


학습 단계 플래그를 반환한다.

학습 단계 플래그는 bool 텐서이며 (0 = 테스트, 1 = 훈련),
훈련 시점과 테스트 시점의 동작 방식이 다른 케라스 함수에
입력으로 준다.

__반환__

학습 단계 (스칼라 정수 텐서 또는 파이썬 정수).

----

### set_learning_phase


```python
keras.backend.set_learning_phase(value)
```


학습 단계를 어떤 고정 값으로 설정한다.

__인자__

- __value__: 학습 단계 값. 0 또는 1 (정수).

__예외__

- __ValueError__: `value`가 `0`이나 `1`이 아닌 경우.

----

### floatx


```python
keras.backend.floatx()
```


기본 실수 타입을 문자열 형태로 반환한다.
(가령 'float16', 'float32', 'float64'.)

__반환__

문자열. 현재 기본 실수 타입.

__예시__

```python
>>> keras.backend.floatx()
'float32'
```

----

### set_floatx


```python
keras.backend.set_floatx(floatx)
```


기본 실수 타입을 설정한다.

__인자__

- __floatx__: 문자열. 'float16' 또는 'float32' 또는 'float64'.

__예시__

```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> K.set_floatx('float16')
>>> K.floatx()
'float16'
```

----

### cast_to_floatx


```python
keras.backend.cast_to_floatx(x)
```


Numpy 배열을 케라스 기본 실수 타입으로 캐스팅 한다.

__인자__

- __x__: Numpy 배열.

__반환__

새 타입으로 캐스팅 한 같은 Numpy 배열.

__예시__

```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> arr = numpy.array([1.0, 2.0], dtype='float64')
>>> arr.dtype
dtype('float64')
>>> new_arr = K.cast_to_floatx(arr)
>>> new_arr
array([ 1.,  2.], dtype=float32)
>>> new_arr.dtype
dtype('float32')
```

----

### image_data_format


```python
keras.backend.image_data_format()
```


기본 이미지 데이터 형식 규약('channels_first' 또는 'channels_last')을 반환한다.

__반환__

문자열. `'channels_first'` 또는 `'channels_last'`.

__예시__

```python
>>> keras.backend.image_data_format()
'channels_first'
```

----

### set_image_data_format


```python
keras.backend.set_image_data_format(data_format)
```


데이터 형식 규약 값을 설정한다.

__인자__

- __data_format__: 문자열. `'channels_first'` or `'channels_last'`.

__예시__

```python
>>> from keras import backend as K
>>> K.image_data_format()
'channels_first'
>>> K.set_image_data_format('channels_last')
>>> K.image_data_format()
'channels_last'
```

----

### epsilon


```python
keras.backend.epsilon()
```


수식에 쓰는 퍼징 인자 값을 반환한다.

__반환__

float.

__예시__

```python
>>> keras.backend.epsilon()
1e-07
```

----

### set_epsilon


```python
keras.backend.set_epsilon(e)
```


수식에 쓰는 퍼징 인자 값을 설정한다.

__인자__

- __e__: float. 새 엡실론 값.

__예시__

```python
>>> from keras import backend as K
>>> K.epsilon()
1e-07
>>> K.set_epsilon(1e-05)
>>> K.epsilon()
1e-05
```






