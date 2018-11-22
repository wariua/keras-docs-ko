# 케라스 FAQ: 자주 묻는 케라스에 대한 질문들

- [케라스를 어떻게 인용하면 되나?](#_1)
- [케라스를 GPU에서 돌리려면?](#gpu)
- [케라스 모델을 여러 GPU에서 돌리려면?](#gpu_1)
- ["표본", "배치", "에포크"가 무슨 뜻인가?](#_4)
- [케라스 모델을 저장하려면?](#_5)
- [왜 훈련 때 손실이 테스트 때 손실보다 훨씬 높은가?](#_10)
- [중간 층의 출력을 얻으려면?](#_11)
- [메모리에 다 안 들어가는 데이터셋에 케라스를 쓰려면?](#_12)
- [검증 손실이 더는 줄지 않을 때 훈련을 중단하려면?](#_13)
- [검증 몫을 어떻게 계산하는가?](#_14)
- [훈련 동안 데이터를 뒤섞는가?](#_15)
- [에포크마다 훈련/검증 손실/정확도를 기록하려면?](#_16)
- [케라스 층을 "얼려" 두려면?](#_17)
- [상태 유지형 RNN을 쓰려면?](#rnn)
- [Sequential 모델에서 층을 빼려면?](#sequential)
- [사전 훈련 모델을 케라스에서 쓰려면?](#_18)
- [케라스에서 HDF5 입력을 쓰려면?](#hdf5)
- [케라스 설정 파일이 어디에 저장되는가?](#_19)
- [개발 과정에서 케라스로 재현 가능한 결과를 얻으려면?](#_20)
- [케라스에서 모델을 저장하기 위해 HDF5 내지 h5py를 설치하려면?](#hdf5-h5py)

---

### 케라스를 어떻게 인용하면 되나?

연구에 도움이 됐다면 출판물에서 케라스를 인용해 달라. 다음은 BibTeX 항목 예이다.

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}
```

---

### 케라스를 GPU에서 돌리려면?

**텐서플로우**나 **CNTK** 백엔드에서 돌리고 있다면 사용 가능한 GPU 탐지 시 자동으로 GPU에서 코드가 돈다.

**테아노** 백엔드에서 돌리고 있다면 다음 중 한 방법을 쓸 수 있다.

**방법 1**: 테아노 플래그 사용.
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

장치 식별자에 따라 'gpu'라는 이름을 (가령 "gpu0", "gpu1" 등으로) 바꿔 줘야 할 수도 있다.

**방법 2**: `.theanorc` 설정하기: [설정 방법](http://deeplearning.net/software/theano/library/config.html)

**방법 3**: 코드 시작에서 수동으로 `theano.config.device`와 `theano.config.floatX` 설정하기.
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### 케라스 모델을 여러 GPU에서 돌리려면?

**텐서플로우** 백엔드를 써서 그렇게 하는 걸 권장한다. 한 모델을 여러 GPU에서 돌리는 데는 두 가지 방법이 있는데, **데이터 병렬화**와 **장치 병렬화**다.

대부분의 경우에선 분명 데이터 병렬화가 필요할 것이다.

#### 데이터 병렬화

데이터 병렬화란 대상 모델을 각 장치로 복제하고서 각 복제본이 입력 데이터의 다른 부분을 처리하게 하는 것이다.
케라스에 내장된 유틸리티인 `keras.utils.multi_gpu_model`을 쓰면 어떤 모델이든 데이터 병렬 버전을 만들어 낼 수 있으며 8개까지 GPU에서 거의 선형적인 속도 향상을 이룰 수 있다.

자세한 내용은 [multi_gpu_model](/utils/#multi_gpu_model) 설명을 보라. 다음은 간단한 예이다.

```python
from keras.utils import multi_gpu_model

# `model`을 8개 GPU 상으로 복제.
# 머신에 사용 가능한 GPU가 8개 있다고 가정.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# 이 `fit` 호출이 8개 GPU로 분산됨.
# 배치 크기가 256이므로 각 GPU가 32개 표본씩 처리하게 됨.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

#### 장치 병렬화

장치 병렬화는 동일 모델의 다른 부분들을 다른 장치에서 돌리는 것이다. 병렬 구조인 모델(가령 두 개 분기 경로가 있는 모델)에서 잘 동작한다.

텐서플로우 장치 스코프를 이용하면 가능하다. 다음은 간단한 예이다.

```python
# 공유 LSTM을 사용해 두 가지 열을 병렬로 인코딩 하는 모델
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 한 GPU에서 첫 번째 열 처리
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# 다른 GPU에서 다음 열 처리
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# CPU에서 결과 이어 붙이기
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

---

### "표본", "배치", "에포크"가 무슨 뜻인가?

다음은 케라스를 올바로 활용하기 위해 알아 둬야 할 몇 가지 정의들이다.

- **표본(sample)**: 데이터셋의 한 항목.
  - *예*: 합성곱 망에선 이미지 하나가 **표본**이다.
  - *예*: 음성 인식 모델에선 음성 파일 하나가 **표본**이다.
- **배치(batch)**: *N*개 표본으로 된 집합. 한 **배치** 내의 표본들은 독립적으로 병렬로 처리된다. 훈련인 경우 배치 하나로 모델에 갱신이 한 번 이뤄진다.
  - **배치**는 일반적으로 단일 입력보다 입력 데이터의 분포에 더 가깝다. 배치가 클수록 더 가깝다. 하지만 배치를 처리하는 데 더 오랜 시간이 걸리고 그런데도 한 번의 갱신만 이뤄진다는 것 역시 사실이다. 추론(평가/예측)에서는 메모리를 바닥내지 않는 한 가능하면 큰 배치 크기를 택하는 걸 권장한다. (배치가 크면 일반적으로 평가/예측이 더 빨라지기 때문이다.)
- **에포크(epoch)**: 훈련을 구별되는 단계들로 분리하기 위해 임의로 나눈 것이며, 일반적으로 "데이터셋 전체 한 번 돌기"로 정의된다. 로깅이나 주기적 평가에 쓸모가 있다.
  - 케라스 모델의 `fit` 메소드에 `evaluation_data`와 `evaluation_split`을 쓰면 **에포크** 끝마다 평가를 실행하게 된다.
  - 케라스에선 특별히 **에포크** 끝에서 돌게 만들어진 [콜백](https://keras.io/callbacks/)을 추가하는 게 가능하다. 예로 학습 속도 변경이나 모델 체크 포인트(저장) 등이 있다.

---

### 케라스 모델을 저장하려면?

#### 모델 전체(구조 + 가중치 + 옵티마이저 상태) 저장하기/불러오기

*케라스 모델 저장에 pickle이나 cPickle을 쓰는 걸 권장하지 않는다.*

`model.save(filepath)`를 사용해 케라스 모델을 HDF5 파일 하나로 저장할 수 있다. 파일에 다음 내용이 담긴다.

- 모델의 구조. 이를 통해 모델 재생성이 가능.
- 모델의 가중치들
- 훈련 설정 (손실, 옵티마이저)
- 옵티마이저 상태. 이를 통해 정확히 멈췄던 지점부터 훈련을 재개할 수 있음.

그러고 나면 `keras.models.load_model(filepath)`을 사용해 모델을 다시 생성할 수 있다.
`load_model`에서 저장된 훈련 설정으로 모델 컴파일까지 해 준다.
(모델이 컴파일 된 적이 있다면 말이다.)

`h5py`를 설치하는 방법에 대해선 [케라스에서 모델을 저장하기 위해 HDF5 내지 h5py를 설치하려면?](#hdf5-h5py) 항목을 보라.

예:

```python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```

#### 모델의 구조만 저장하기/불러오기

**모델의 구조**만 저장하면 되고 가중치나 훈련 설정은 필요치 않다면 다음처럼 할 수 있다.

```python
# JSON으로 저장
json_string = model.to_json()

# YAML로 저장
yaml_string = model.to_yaml()
```

생성되는 JSON/YAML 파일은 사람이 읽을 수 있는 형식이고 필요 시 수동으로 편집할 수 있다.

그러고 나면 이 데이터를 가지고 신선한 모델을 구축할 수 있다.

```python
# JSON에서 가져와서 모델 재구성
from keras.models import model_from_json
model = model_from_json(json_string)

# YAML에서 가져와서 모델 재구성
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### 모델의 가중치만 저장하기/불러오기

**모델의 가중치**만 저장하면 된다면 HDF5에서 아래 코드처럼 하면 된다.

```python
model.save_weights('my_model_weights.h5')
```

모델 인스턴스를 만드는 코드가 있다고 하고, 그러면 저장했던 가중치를 *같은* 구조의 모델로 적재할 수 있다.

```python
model.load_weights('my_model_weights.h5')
```

미세 조정이나 이전 학습(transfer-learning) 등을 위해 (일부 층이 공통인) *다른* 구조로 가중치를 적재해야 한다면 *층 이름*으로 가중치를 적재할 수 있다.

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

`h5py`를 설치하는 방법에 대해선 [케라스에서 모델을 저장하기 위해 HDF5 내지 h5py를 설치하려면?](#hdf5-h5py) 항목을 보라.

예:

```python
"""
원래 모델이 다음과 같다고 하자:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# 새 모델
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 적재됨
model.add(Dense(10, name='new_dense'))  # 적재되지 않음

# 첫 번째 모델의 가중치 적재하기. 첫 번째 층 dense_1만 영향을 받음
model.load_weights(fname, by_name=True)
```

#### 저장된 모델의 자체 제작 층(또는 다른 자체 제작 객체) 다루기

적재하려는 모델에 자체 제작 층이나 여타 자체 제작 클래스 내지 함수가 있다면
`custom_objects` 인자를 통해 적재 메커니즘에게 전달해 줄 수 있다.

```python
from keras.models import load_model
# 모델에 "AttentionLayer" 클래스의 인스턴스가 있다고 하자
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

아니면 [자체 제작 객체 스코프](https://keras.io/utils/#customobjectscope)를 쓸 수도 있다.

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

자체 제작 객체 처리 방식은 `load_model`, `model_from_json`, `model_from_yaml`에서 모두 동일하다.

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 왜 훈련 때 손실이 테스트 때 손실보다 훨씬 높은가?

케라스 모델에는 훈련 모드와 테스트 모드가 있다. 테스트 때는 드롭아웃이나 L1/L2 가중치 정칙화 같은 정칙화 메커니즘들을 끈다.

더불어 훈련 손실은 각 훈련 데이터 배치에 대한 손실의 평균이다. 시간이 지나면서 모델이 바뀌기 때문에 에포크 첫 번째 배치에 대한 손실이 일반적으로 마지막 배치보다 높다. 한편으로 에포크의 테스트 손실은 에포크 마지막의 모델을 이용해 계산하고, 그래서 낮은 손실이 나온다.

---

### 중간 층의 출력을 얻으려면?

간단한 방법 하나는 관심 있는 층을 출력하는 새 `Model`을 만드는 것이다.

```python
from keras.models import Model

model = ...  # 원래 모델 만들기

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

아니면 어떤 입력을 주면 특정 층의 출력을 반환하는 케라스 함수를 만들 수도 있다. 예:

```python
from keras import backend as K

# Sequential 모델에서
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```

이와 비슷하게 테아노와 텐서플로우의 함수를 직접 만들 수도 있다.

참고로 모델이 훈련 단계와 테스트 단계에서 다르게 동작한다면 (가령 `Dropout`, `BatchNormalization` 등을 쓴다면)
함수에 학습 단계 플래그를 줘야 할 것이다.

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# 테스트 모드(0)에서의 출력
layer_output = get_3rd_layer_output([x, 0])[0]

# 훈련 모드(1)에서의 출력
layer_output = get_3rd_layer_output([x, 1])[0]
```

---

### 메모리에 다 안 들어가는 데이터셋에 케라스를 쓰려면?

`model.train_on_batch(x, y)` 및 `model.test_on_batch(x, y)`를 이용해 배치 훈련을 할 수 있다. [모델 설명](/models/sequential) 참고.

또는 훈련 데이터 배치를 내놓는 제너레이터를 작성하고 `model.fit_generator(data_generator, steps_per_epoch, epochs)` 메소드를 쓸 수도 있다.

[CIFAR10 예시](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)에서 실제 동작하는 배치 훈련을 볼 수 있다.

---

### 검증 손실이 더는 줄지 않을 때 훈련을 중단하려면?

`EarlyStopping` 콜백을 쓸 수 있다.

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

자세한 건 [콜백 설명](/callbacks)을 보라.

---

### 검증 몫을 어떻게 계산하는가?

`model.fit`의 `validation_split` 인자를 가령 0.1로 설정하면 데이터의 *마지막 10%*를 검증 데이터로 쓰게 된다. 그리고 0.25로 설정하면 데이터의 마지막 25%를 쓰게 되는 식이다. 참고로 검증 몫을 빼내기 전에 데이터를 뒤섞지 않으며, 그래서 입력해 준 표본에서 말 그대로 *마지막* x%가 검증 데이터이다.

동일한 검증 세트를 (같은 `fit` 호출 내의) 모든 에포크에 사용한다.

---

### 훈련 동안 데이터를 뒤섞는가?

그렇다. `model.fit`의 `shuffle` 인자가 `True`(기본값)로 설정돼 있으면 에포크마다 훈련 데이터를 난수적으로 뒤섞는다.

검증 데이터는 절대 뒤섞지 않는다.

---


### 에포크마다 훈련/검증 손실/정확도를 기록하려면?

`model.fit` 메소드가 `History` 콜백을 반환하는데 그 `history` 속성이 연속된 손실 및 기타 지표들의 목록을 담고 있다.

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### 케라스 층을 "얼려" 두려면?

층을 "얼린다"는 건 훈련에서 배제한다는 뜻이다. 즉 가중치가 절대 갱신되지 않게 된다. 모델을 미세 조정하거나 텍스트 입력에 고정 embedding을 쓸 때 유용하다.

층 생성자에 `trainable` 인자(불리언)를 줘서 층에 훈련 불가 설정을 할 수 있다.

```python
frozen_layer = Dense(32, trainable=False)
```

또 인스턴스 생성 후에도 층의 `trainable` 속성을 `True`나 `False`로 설정할 수 있다. 설정이 효력이 있으려면 `trainable` 속성 변경 후 모델에 `compile()`을 호출해 줘야 한다. 예:

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# 아래 모델에서 `layer`의 가중치가 훈련 동안 갱신되지 않음
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# 이 모델에서는 `layer`의 가중치가 훈련 동안 갱신됨
# (같은 층 인스턴스를 쓰는 위 모델에도 영향을 주게 됨)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # `layer`의 가중치를 갱신하지 않음
trainable_model.fit(data, labels)  # `layer`의 가중치를 갱신함
```

---

### 상태 유지형 RNN을 쓰려면?

RNN을 상태 유지형으로 만든다는 건 각 배치의 표본들에 대한 상태를 다음 배치 표본들을 위한 초기 상태로 재사용하게 한다는 뜻이다.

따라서 상태 유지형 RNN을 쓸 때는 다음을 상정한다.

- 모든 배치의 표본 수가 같다.
- `x1`과 `x2`가 연속한 표본 배치일 때 모든 `i`에 대해 `x2[i]`가 `x1[i]`의 후속 열이다.

RNN에서 상태 유지 방식을 쓰려면:

- 모델 첫 번째 층에 `batch_size` 인자를 줘서 사용할 배치 크기를 명시적으로 지정해야 한다. 가령 timestep당 16개 피쳐가 있는 10개 timestep 열들의 32개 표본 배치에 대해 `batch_size=32`.
- RNN 층(들)에 `stateful=True`를 설정해야 한다.
- `fit()` 호출 시 `shuffle=False`를 지정해 줘야 한다.

누적된 상태를 재설정하려면:

- `model.reset_states()`로 모델 내 모든 층들의 상태를 재설정
- `layer.reset_states()`로 상태 유지형 RNN의 특정 층의 상태를 재설정

예:

```python
x  # 입력 데이터, (32, 21, 16) 형태
# 이를 길이 10짜리 열로 모델에 넣어 줄 것이다.

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 처음 10개 timestep을 가지고 11번째 timestep을 예측하도록 망을 훈련
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 망의 상태가 바뀌었다. 후속 열을 넣어 줄 수 있다.
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# LSTM 층의 상태를 재설정
model.reset_states()

# 이 경우 가능한 다른 방식
model.layers[0].reset_states()
```

참고로 메소드 `predict`, `fit`, `train_on_batch`, `predict_classes` 등이 *모두* 모델의 상태 유지형 층의 상태를 갱신하게 된다. 그래서 상태 기반 훈련뿐 아니라 상태 기반 예측도 할 수 있다.

---

### Sequential 모델에서 층을 빼려면?

Sequential 모델에서 `.pop()`을 호출하면 마지막으로 추가한 층을 제거할 수 있다.

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### 사전 훈련 모델을 케라스에서 쓰려면?

다음 이미지 분류 모델에 대한 코드와 사전 훈련된 가중치를 바로 쓸 수 있다.

- Xception
- VGG16
- VGG19
- ResNet50
- Inception v3
- Inception-ResNet v2
- MobileNet v1

`keras.applications` 모듈에서 가져올 수 있다.

```python
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet

model = VGG16(weights='imagenet', include_top=True)
```

몇 가지 간단한 사용 예를 [Applications 모듈 설명](/applications)에서 볼 수 있다.

피처 추출이나 미세 조정을 위해 사전 훈련 모델을 사용하는 방법에 대한 자세한 예를 [이 블로그 글](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)에서 볼 수 있다.

VGG16 모델은 여러 케라스 예시 스크립트의 기반이기도 하다.

- [스타일 전환](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
- [피쳐 시각화](https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep Dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)

---

### 케라스에서 HDF5 입력을 쓰려면?

`keras.utils.io_utils`에 있는 `HDF5Matrix`를 쓸 수 있다. 자세한 내용은 [HDF5Matrix 설명](/utils/#hdf5matrix)을 보라.

HDF5 데이터셋을 직접 사용할 수도 있다.

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

`h5py`를 설치하는 방법에 대해선 [케라스에서 모델을 저장하기 위해 HDF5 내지 h5py를 설치하려면?](#hdf5-h5py) 항목을 보라.

---

### 케라스 설정 파일이 어디에 저장되는가?

모든 케라스 데이터가 저장되는 기본 디렉터리는 다음 위치이다.

```bash
$HOME/.keras/
```

참고로 윈도우 사용자는 `$HOME`을 `%USERPROFILE%`로 바꿔 주면 된다.
케라스에서 (가령 권한 문제 등으로) 위 디렉터리를 만들 수 없는 경우에는 백업으로 `/tmp/.keras/`를 쓴다.

케라스 설정 파일은 JSON 파일이며 `$HOME/.keras/keras.json`에 저장된다. 기본 설정 파일은 다음과 같다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

다음 필드를 담고 있다.

- 이미지 처리 층과 유틸리티에서 기본으로 사용할 이미지 데이터 형식 (`channels_last` 또는 `channels_first`)
- 일부 연산에서 0으로 나누기를 방지하기 위해 사용하는 수치 fuzz 인자 `epsilon`.
- 기본 부동소수점 데이터 타입.
- 기본 백엔드. [백엔드 설명](/backend) 참고.

마찬가지로 [`get_file()`](/utils/#get_file)로 내려받은 것 같은 캐싱 된 데이터셋 파일들을 기본적으로 `$HOME/.keras/datasets/`에 저장한다.

---

### 개발 과정에서 케라스로 재현 가능한 결과를 얻으려면?

모델 개발 과정에서 성능 변화가 실제 모델 내지 데이터 변경 때문인지 아니면 새 무작위 표본의 결과일 뿐인지 판단하기 위해서 돌릴 때마다 재현 가능한 결과를 얻을 수 있으면 좋은 경우가 가끔씩 있다.

일단 프로그램 시작 전에 (프로그램 내에서가 아님) 환경 변수 `PYTHONHASHSEED`를 `0`으로 설정해야 한다. 파이썬 3.2.3 및 이후에서 특정 해시 기반 동작들(가령 set나 dict의 항목 순서. 자세한 건 [파이썬 문서](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED)나 [이슈 #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926) 참고)이 재현 가능하도록 하려면 필요하다. 환경 변수를 설정하는 한 방법은 다음처럼 파이썬 시작 때 설정하는 것이다.

```
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # 재현 불가능한 해시 (파이썬 3.2.3+)
-8127205062320133199
$ python3 test_hash.py                  # 재현 불가능한 해시 (파이썬 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # 재현 가능한 해시
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # 재현 가능한 해시
4883664951434749476
```

더불어 백엔드로 텐서플로우를 써서 GPU에서 돌릴 때 일부 연산의 출력이 비결정적인데, 특히 `tf.reduce_sum()`이 그렇다. GPU에서 여러 연산을 병렬로 돌리기 때문에 실행 순서가 항상 보장되지는 않기 때문이다. 또 부동소수점의 제한된 정밀도 때문에 숫자 몇 개를 더할 때조차도 더하는 순서에 따라 살짝 다른 결과가 나올 수 있다. 비결정적 연산을 피하려고 시도할 수는 있지만 일부는 텐서플로우에서 경사 계산을 위해 자동으로 생성할 수도 있다. 따라서 코드를 CPU에서만 돌리는 게 훨씬 간단하다. 그러기 위해 환경 변수 `CUDA_VISIBLE_DEVICES`를 빈 문자열로 설정할 수 있다. 예:

```
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
```

아래 코드에서 재현 가능한 결과를 얻는 방법의 예를 볼 수 있다. 텐서플로우 백엔드와 파이썬 3 환경에 맞춰져 있다.

```python
import numpy as np
import tensorflow as tf
import random as rn

# Numpy에서 생성하는 난수를 잘 정의된 초기 상태로 시작하기 위해
# 아래 코드가 필요하다.

np.random.seed(42)

# 파이썬 코어에서 생성하는 난수를 잘 정의된 초기 상태로 시작하기
# 위해 아래 코드가 필요하다.

rn.seed(12345)

# 텐서플로우에서 단일 스레드를 쓰게 한다.
# 다중 스레드는 재현 불가능한 결과의 잠재적 원천이다.
# 자세한 내용은 다음 참고:
# https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# 아래의 tf.set_random_seed()는 텐서플로우 백엔드의 난수 생성
# 기능을 잘 정의된 초기 상태로 만든다.
# 자세한 내용은 다음 참고:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# 코드 나머지 부분...
```

---

### 케라스에서 모델을 저장하기 위해 HDF5 내지 h5py를 설치하려면?

`keras.callbacks.ModelCheckpoint` 등을 통해 케라스 모델을
HDF5 파일로 저장하기 위해 케라스에서는 파이썬 패키지 h5py를 사용한다.
케라스에 필수인 패키지이므로 기본으로 설치될 것이다.
데비안 기반 배포판에서는 `libhdf5`를 추가로 설치해 줘야 할 것이다.

```
sudo apt-get install libhdf5-serial-dev
```

h5py가 설치돼 있는지 잘 모르겠으면 파이썬 셸을 열고 모듈을 올려 보면 된다.

```
import h5py
```

오류 없이 가져올 수 있으면 설치된 것이고, 아니라면 다음에서 자세한
설치 방법을 볼 수 있다: http://docs.h5py.org/en/latest/build.html
