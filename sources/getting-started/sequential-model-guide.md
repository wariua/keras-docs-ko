# 케라스 Sequential 모델 써 보기

`Sequential` 모델은 층을 선형으로 쌓은 것이다.

생성자에 층 인스턴스 목록을 줘서 `Sequential` 모델을 만들 수 있다.

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

`.add()` 메소드를 통해 층을 간단히 추가할 수도 있다.

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

----

## 입력 형태 지정하기

모델에서는 기대하는 입력 형태에 대해 알고 있어야 한다. 그래서 `Sequential` 모델의 첫 번째 층에서 입력 형태에 대한 정보를 받아야 한다. (첫 번째 층에서만이다. 이어지는 층들에서는 자동으로 형태를 추론할 수 있다.) 이를 위한 방법이 여러 가지 있다.

- 첫 번째 층에 `input_shape` 인자 주기. 이 인자는 형태 튜플(정수 또는 `None` 항목으로 이뤄진 튜플. `None`은 기대할 수 있는 모든 양의 정수를 나타냄)이다. `input_shape`에 배치 차원은 포함되지 않는다.
- `Dense` 같은 일부 2D 층에선 인자 `input_dim`을 통한 입력 형태 지정을 지원하며 어떤 3D temporal 층에선 `input_dim` 및 `input_length` 인자를 지원한다.
- 만약 고정된 배치 크기를 입력으로 지정해야 한다면 (상태 유지형 순환 망에서 유용함) `batch_size` 인자를 층에 줄 수 있다. 층에 `batch_size=32`와 `input_shape=(6, 8)`을 같이 주면 모든 입력 배치가 배치 형태 `(32, 6, 8)`이라고 기대하게 된다.

그래서 다음 두 코드는 정확하게 동등하다.
```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
```
```python
model = Sequential()
model.add(Dense(32, input_dim=784))
```

----

## 컴파일

모델을 훈련시키기 전에 학습 과정을 설정해야 하는데, `compile` 메소드를 통해 이뤄진다. 세 개 인자를 받는다.

- 옵티마이저. 기존 옵티마이저의 문자열 식별자(`rmsprop`나 `adagrad` 등)일 수도 있고 `Optimizer` 클래스의 인스턴스일 수도 있다. [옵티마이저](/optimizers) 참고.
- 손실 함수. 모델에서 최소화하려 하는 목표이다. 기존 손실 함수의 문자열 식별자(`categorical_crossentropy`나 `mse` 등)일 수도 있고 목표 함수일 수도 있다. [손실](/losses) 참고.
- 측정 방식 목록. 분류 문제라면 `metrics=['accuracy']`라고 설정하고 싶을 것이다. 기존 측정 방식의 문자열 식별자일 수도 있고 따로 만든 측정 함수일 수도 있다.

```python
# 다중 분류 문제
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 이진 분류 문제
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 평균 제곱 오차 회귀 문제
model.compile(optimizer='rmsprop',
              loss='mse')

# 자체 측정 방식
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

----

## 훈련

입력 데이터와 레이블의 Numpy 배열을 가지고 케라스 모델을 훈련시킨다. 모델 훈련에는 보통 `fit` 함수를 쓰게 된다. [여기서 설명한다](/models/sequential).

```python
# 2개 유형 (이진 분류) 단일 입력 모델:

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 더미 데이터 생성
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# 32개 표본 배치로 데이터를 돌며 모델 훈련
model.fit(data, labels, epochs=10, batch_size=32)
```

```python
# 10개 유형 (범주 분류) 단일 입력 모델:

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 더미 데이터 생성
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# 레이블을 범주형 원핫 인코딩으로 변환
# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# 32개 표본 배치로 데이터를 돌며 모델 훈련
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

----


## 예시

살펴볼 만한 예시들이 몇 가지 있다.

[예시 폴더](https://github.com/keras-team/keras/tree/master/examples)에서 실제 데이터셋에 대한 예시 모델들을 볼 수 있다.

- CIFAR10 작은 이미 분류: 실시간 데이터 증대를 하는 합성곱 신경망(CNN)
- IMDB 영화 감상평 감정 분류: 단어 열에 대한 LSTM
- 로이터 뉴스 서비스 주제 분류: 다층 퍼셉트론(MLP)
- MNIST 필기 숫자 분류: MLP & CNN
- LSTM을 이용한 문자 수준 텍스트 생성

그 외 여러 가지가 있다.


### 다유형 소프트맥스 분류를 위한 다층 퍼셉트론(MLP):

```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# 더미 데이터 생성
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64)는 64개 은닉 유닛을 가진 완전 연결 층이다.
# 첫 번째 층에서 예상 입력 데이터 형태를 지정해야 하는데,
# 여기선 20차원 벡터이다.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```


### 이진 분류를 위한 MLP:

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 더미 데이터 생성
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```


### VGG 비슷한 합성곱 망:

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 더미 데이터 생성
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# 입력: 3채널 100x100 이미지 -> (100, 100, 3) 텐서.
# 각기 3x3인 합성곱 필터 32개 적용.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```


### LSTM을 이용한 순차 분류:

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

max_features = 1024

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 1D 합성곱을 이용한 순차 분류:

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

seq_length = 64

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 순차 분류를 위한 중첩 LSTM

이 모델에서는 LSTM 층 3개를 쌓아서
고수준 temporal 표현을 학습할 수 있는 모델을 만든다.

처음 두 LSTM은 출력 열 전체를 반환하지만 마지막 LSTM은
출력 열의 마지막 단계만 반환해서 temporal 차원을 버린다.
(즉 입력 열을 벡터 한 개로 변환한다.)

<img src="https://keras.io/img/regular_stacked_lstm.png" alt="stacked LSTM" style="width: 300px;"/>

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# 예상 입력 데이터 형태: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # 32차원 벡터 열 반환
model.add(LSTM(32, return_sequences=True))  # 32차원 벡터 열 반환
model.add(LSTM(32))  # 32차원 벡터 1개 반환
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 더미 훈련 데이터 생성
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# 더미 검사 데이터 생성
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
```


### 동일한 중첩 LSTM 모델, "상태 유지형"으로

상태 유지형 순환 모델에서는 표본 배치를 하나 처리하고서 얻은 내부 상태(메모리)를
다음 표본 배치를 위한 초기 상태로 재사용한다. 이렇게 하면 연산 복잡도를
관리 가능한 수준으로 유지하면서 더 긴 열을 처리할 수 있다.

[FAQ에 상태 유지형 RNN에 대한 내용이 더 있다.](/getting-started/faq/#how-can-i-use-stateful-rnns)

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# 예상 입력 배치 형태: (batch_size, timesteps, data_dim)
# 망이 상태 유지형이므로 batch_input_shape 전체를 제공해 줘야 한다.
# k번째 배치의 i번째 표본이 k-1번째 배치의 i번 표본의 후속 항목이다.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 더미 훈련 데이터 생성
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# 더미 검사 데이터 생성
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
```
