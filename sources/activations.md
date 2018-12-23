
## 활성 사용

`Activation` 층을 통해서, 또는 모든 전달 층에서 지원하는 `activation` 인자를 통해서 활성 함수를 사용할 수 있다.

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

이는 다음과 동등하다.

```python
model.add(Dense(64, activation='tanh'))
```

항목 단위 텐서플로우/테아노/CNTK 함수를 활성으로 줄 수도 있다.

```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
```

## 사용 가능한 활성

### softmax


```python
keras.activations.softmax(x, axis=-1)
```


소프트맥스 활성 함수.

__인자__

- __x__: 입력 텐서.
- __axis__: 정수. 소프트맥스 정규화를 적용할 축.

__반환__

텐서. 소프트맥스 변환의 출력.

__예외__

- __ValueError__: `dim(x) == 1`인 경우.

----

### elu


```python
keras.activations.elu(x, alpha=1.0)
```


지수 선형 단위(exponential linear unit).

__인자__

- __x__: 입력 텐서.
- __alpha__: 스칼라. 음수 구역에서의 기울기.

__반환__

지수 선형 활성: `x > 0`이면 `x`이고,
`x < 0`이면 `alpha * (exp(x)-1)`.

__참고 자료__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

----

### selu


```python
keras.activations.selu(x)
```


조정 지수 선형 단위(Scaled Exponential Linear Unit, SELU).

SELU는 `scale * elu(x, alpha)`와 같다. 여기서 alpha와 scale은
미리 정의된 상수다. 가중치들을 올바로 (`lecun_normal` 초기화 참고)
초기화 하고 입력 수가 "충분히 많으면" (자세한 내용은 참고 자료를 보라)
입력의 평균과 분산이 유지되도록 `alpha`와 `scale`의 값이 정해져 있다.

__인자__

- __x__: 활성 함수를 계산할 텐서 내지 변수.

__반환__

조정 지수 단위 활성: `scale * elu(x, alpha)`.

__주의__

- 초기화 "lecun_normal"과 함께 사용해야 함.
- 드롭아웃 방식 "AlphaDropout"과 함께 사용해야 함.

__참고 자료__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

----

### softplus


```python
keras.activations.softplus(x)
```


softplus 활성 함수.

__인자__

- __x__: 입력 텐서.

__반환__

softplus 활성: `log(exp(x) + 1)`.

----

### softsign


```python
keras.activations.softsign(x)
```


softsign 활성 함수.

__인자__

- __x__: 입력 텐서.

__반환__

softplus 활성: `x / (abs(x) + 1)`.

----

### relu


```python
keras.activations.relu(x, alpha=0.0, max_value=None)
```


정류 선형 단위(Rectified Linear Unit).

__인자__

- __x__: 입력 텐서.
- __alpha__: 음수 부분에서의 기울기. 기본은 0.
- __max_value__: 출력의 최댓값.

__반환__

(누출형) 정류 선형 단위 활성: `x > 0`이면 `x`이고,
`x < 0`이면 `alpha * x`. `max_value`가 정의돼 있으면
결과를 그 값으로 잘라낸다.

----

### tanh


```python
keras.activations.tanh(x)
```


쌍곡탄젠트 활성 함수.

----

### sigmoid


```python
keras.activations.sigmoid(x)
```


시그모이드(S자 모양) 활성 함수.

----

### hard_sigmoid


```python
keras.activations.hard_sigmoid(x)
```


하드 시그모이드 활성 함수.

시그모이드 활성보다 계산이 빠름.

__인자__

- __x__: 입력 텐서.

__반환__

하드 시그모이드 활성:

- `x < -2.5`이면 `0`
- `x > 2.5`이면 `1`
- `-2.5 <= x <= 2.5`이면 `0.2 * x + 0.5`

----

### linear


```python
keras.activations.linear(x)
```


선형 (즉 항등) 활성 함수.


## "고급 활성 함수"에 대해

단순한 텐서플로우/테아노/CNTK 함수 이상으로 복잡한 활성 함수들(가령 상태를 유지하는 학습 가능한 활성)은 [고급 활성 층](layers/advanced-activations.md)에 있으며 `keras.layers.advanced_activations` 모듈에서 볼 수 있다. `PReLU`와 `LeakyReLU` 등이 있다.
