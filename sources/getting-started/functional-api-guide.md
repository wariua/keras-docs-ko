# 케라스 함수형 API 써 보기

케라스 함수형 API를 이용하면 다출력 모델, 유향 무순환 그래프, 공유 층 포함 모델 같은 복잡한 모델을 정의할 수 있다.

이 소개에서는 독자가 `Sequential` 모델에 이미 친숙하다고 가정한다.

일단 간단한 걸로 시작해 보자.

-----

## 첫 번째 예: 촘촘히 연결된 망

그런 망을 구현하는 데는 `Sequential` 모델이 더 나은 선택이겠지만 아주 간단히 시작하기에는 좋다.

- 층 인스턴스는 (텐서에 대해) 호출 가능하며 텐서를 반환한다.
- 그리고 입력 텐서(들)과 출력 텐서(들)을 가지고 `Model`을 정의할 수 있다.
- 그 모델을 케라스 `Sequential` 모델들처럼 훈련시킬 수 있다.

```python
from keras.layers import Input, Dense
from keras.models import Model

# 텐서를 반환함
inputs = Input(shape=(784,))

# 층 인스턴스는 텐서에 대해 호출 가능하고 텐서를 반환한다
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Input 층과 세 개의 Dense 층을 포함하는 모델을 만든다
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # 훈련 시작
```

-----

## 모든 모델은 층과 마찬가지로 호출 가능하다

함수형 API를 쓰면 훈련한 모델을 쉽게 재사용할 수 있다. 어떤 모델이든 층처럼 다뤄서 텐서에 대해 호출할 수 있다. 참고로 모델을 호출하면 모델의 *구조*만 재사용하는 게 아니라 그 가중치까지 재사용하게 된다.

```python
x = Input(shape=(784,))
# 이렇게 쓸 수 있으며, 위에서 정의한 10짜리 소프트맥스를 반환한다.
y = model(x)
```

이를 이용하면 예를 들어 입력의 *열*을 처리할 수 있는 모델을 빨리 만들 수 있다. 한 줄만으로 이미지 분류 모델을 영상 분류 모델로 바꿀 수 있을 것이다.

```python
from keras.layers import TimeDistributed

# 20개 timestep 열에 대한 입력 텐서
# 각각이 784차원 벡터를 담고 있음
input_sequences = Input(shape=(20, 784))

# 입력 열의 각 timestep마다 이전 모델을 적용한다.
# 이전 모델의 출력이 10짜리 소프트맥스였으므로
# 아래 층의 출력은 크기 10인 벡터 20개의 열이 될 것이다.
processed_sequences = TimeDistributed(model)(input_sequences)
```

-----

## 다입력 다출력 모델

함수형 API의 좋은 사용 사례가 있는데, 바로 입력과 출력이 여러 개인 모델이다. 함수형 API를 쓰면 많은 뒤얽힌 데이터 스트림들을 조작하는 게 쉬워진다.

다음과 같은 모델을 생각해 보자. 트위터에서 뉴스 헤드라인이 얼마나 많은 리트윗과 좋아요를 받을지 예측하려고 한다. 모델의 주된 입력은 단어들의 열인 헤드라인 자체가 될 것이다. 하지만 양념을 더하기 위해 모델에 부가 입력이 있어서 헤드라인이 올라온 시간 등과 같은 추가 데이터를 받는다.
또한 두 가지 손실 함수를 통해 모델을 지도하게 된다. 주된 손실 함수를 모델 초반에 사용하는 게 심층 모델에 좋은 정규화 방법이다.

모델의 모양은 다음과 같다.

<img src="https://s3.amazonaws.com/keras.io/img/multi-input-multi-output-graph.png" alt="multi-input-multi-output-graph" style="width: 400px;"/>

그럼 이걸 함수형 API로 구현해 보자.

주 입력은 정수들의 열로 된 헤드라인을 받게 된다. (각 정수는 단어를 나타낸다.) 정수는 1에서 10,000 사이이고 (10,000 단어 어휘) 열은 100 단어 길이이다.

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# 헤드라인 입력: 1에서 10000 사이의 정수 100개 열을 받음.
# 참고로 어떤 층이든 "name" 인자로 이름을 붙일 수 있다.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# 이 embedding 층이 입력 열을 밀집 512차원 벡터들의 열로
# 인코딩 하게 된다.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM이 벡터들의 열을 열 전체에 대한 정보를 담은 벡터
# 한 개로 변환하게 된다.
lstm_out = LSTM(32)(x)
```

여기에 보조 손실을 집어넣어서 주 손실이 모델 휠씬 위쪽에 있더라도 LSTM 및 Embedding 층이 매끄럽게 훈련되도록 한다.

```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```

이 지점에서 보조 입력을 LSTM 출력과 이어 붙여서 모델에 집어넣는다.

```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 그 위에 밀집 연결 망을 깊게 쌓는다.
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 마지막으로 주 로지스틱 회귀 층을 추가한다.
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```

다음은 두 가지 입력과 두 가지 출력이 있는 모델을 정의한다.

```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```

모델을 컴파일 하고 보조 손실에 가중치 0.2를 준다.
각 출력에 다른 `loss_weights`나 `loss`를 지정하려면 리스트나 딕셔너리를 쓰면 된다.
여기선 `loss` 인자로 손실을 한 개만 줘서 모든 출력에 같은 손실을 쓰도록 한다.

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```

입력 배열 및 출력 배열 리스트를 줘서 모델을 훈련시킬 수 있다.

```python
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
```

입력과 출력에 ("name" 인자를 줘서) 이름이 있으므로
모델을 다음처럼 컴파일 할 수도 있다.

```python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# 그리고 다음으로 훈련:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```

-----

## 공유 층

함수형 API와 잘 맞는 또 다른 경우가 공유 층을 쓰는 모델이다. 먼저 공유 층을 살펴보자.

트윗들로 된 데이터셋을 생각해 보자. 두 트윗이 같은 사람에게서 온 것인지 아닌지를 알려 주는 모델을 만들려고 한다. (그러면 예를 들어 사람들을 트윗 유사성으로 비교할 수 있게 될 것이다.)

이를 위한 한 방법은 두 트윗을 두 벡터로 인코딩 하는 모델을 만들고, 그 벡터들을 이어 붙인 다음 로지스틱 회귀를 추가하는 것이다. 이 모델은 두 트윗의 작성자가 같을 확률을 출력한다. 그러면 양성인 트윗 쌍과 음성인 트윗 쌍들로 모델을 훈련하게 될 것이다.

확률이 대칭이기 때문에 첫 번째 트윗을 인코딩 하는 메커니즘을 두 번째 트윗을 인코딩 하는 데 (가중치 및 모두를) 재사용하는 게 좋다. 그러면 공유 LSTM 층을 사용해 트윗들을 인코딩 하게 된다.

이걸 함수형 API로 만들어 보자. 트윗에 대한 입력으로는 `(280, 256)` 형태의 이진 행렬을 받게 된다. 즉 크기가 256인 벡터 280개의 열인데, 256차원 벡터의 각 차원이 (자주 쓰는 256개 문자로 된 알파벳의) 어느 문자의 존재/부재를 인코딩 한다.

```python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(280, 256))
tweet_b = Input(shape=(280, 256))
```

한 층을 여러 입력에 걸쳐 공유하려면 그 층을 한 번만 만들고서 원하는 대로 여러 입력에 대해 호출하기만 하면 된다.

```python
# 이 층은 입력으로 행렬을 받을 수 있고
# 크기가 64인 벡터를 반환하게 된다.
shared_lstm = LSTM(64)

# 동일한 층 인스턴스를 여러 번
# 재사용할 때  그 층의 가중치들도
# 재사용된다.
# (실질적으로 *동일한* 층이다.)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 이제 두 벡터를 이어 붙일 수 있다:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# 제일 위에 로지스틱 회귀를 추가한다.
predictions = Dense(1, activation='sigmoid')(merged_vector)

# 트윗 입력과 예측을 연결하는
# 훈련 가능한 모델을 정의한다.
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

잠시 멈춰서 공유 층의 출력 내지 출력 형태를 어떻게 읽는지 살펴 보자.

-----

## 층 "노드" 개념

어떤 입력에 대해서 층을 호출할 때마다 새로운 텐서(층의 출력)를 만들게 되고, 또 층에 "노드"를 추가해서 입력 텐서를 출력 텐서로 연결하게 된다. 같은 층을 여러 번 호출 시 그 층은 0, 1, 2... 식으로 번호가 붙은 여러 노드들을 가지게 된다.

케라스 이전 버전에서는 `layer.get_output()`으로 층 인스턴스의 출력 텐서를 얻거나 `layer.output_shape`으로 출력 형태를 얻을 수 있었다. 지금도 그럴 수 있다. (단 `get_output()`이 `output` 속성으로 교체됐다.) 그런데 층이 여러 입력에 연결돼 있다면 어떻게 될까?

층이 한 입력에만 연결돼 있을 때는 혼동의 여지가 없으며 `.output`이 층의 그 한 개 출력을 반환한다.

```python
a = Input(shape=(280, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

하지만 층에 여러 입력이 있으면 그렇지 않다.
```python
a = Input(shape=(280, 256))
b = Input(shape=(280, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```
```
>> AttributeError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```

그렇다고 한다. 다음처럼 하면 된다.

```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

간단하지 않은가?

`input_shape`와 `output_shape` 속성에 대해서도 마찬가지다. 층에 노드가 하나만 있거나 모든 노드의 입력/출력 형태가 같을 때는 "층의 출력/입력 형태"가 잘 정의되며, 그 형태를 `layer.output_shape`/`layer.input_shape`가 반환하게 된다. 하지만 예를 들어 어떤 `Conv2D` 층을 `(32, 32, 3)` 형태의 입력에 적용하고서 또 `(64, 64, 3)` 형태의 입력에 적용한다면 그 층에 입력/출력 형태가 여러 개 있게 되고, 그래서 형태를 가져오려면 소속 노드의 번호를 지정해야 한다.

```python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# 지금까지 입력이 하나이므로 다음이 동작함:
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# 이제 `.input_shape` 속성이 동작하지 않겠지만 다음은 동작함:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```

-----

## 다른 예들

처음 배울 때는 코드 예시가 최고다. 몇 가지를 더 보자.

### Inception 모듈

Inception 구조에 대한 자세한 내용은 [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842) 참고.

```python
from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
```

### 합성곱 층에서의 residual 연결

residual 망에 대한 자세한 내용은 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) 참고.

```python
from keras.layers import Conv2D, Input

# 3채널 256x256 이미지를 위한 입력 텐서
x = Input(shape=(256, 256, 3))
# 출력 채널이 3개(입력 채널과 동일)인 3x3 conv
y = Conv2D(3, (3, 3), padding='same')(x)
# x + y 반환.
z = keras.layers.add([x, y])
```

### 공유 시각 모델

이 모델에서는 동일 이미지 처리 모듈을 두 입력에 재사용해서 두 MNIST 숫자가 같은 숫자인지 아닌지를 분류한다.

```python
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# 먼저 시각 모듈을 정의
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# 그리고서 숫자 구별 모델 정의
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# 시각 모델의 가중치와 모든 부분이 공유됨
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
```

### 시각 질의 응답 모델

이 모델은 그림에 대한 자연 언어 질문을 받았을 때 올바른 한 단어짜리 답을 선택할 수 있다.

질문을 벡터로 인코딩 하고, 이미지를 벡터로 인코딩 하고, 둘을 이어 붙이고, 어떤 답변 후보들에 대해 로지스틱 회귀를 훈련시키는 방식으로 동작한다.

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# 먼저 Sequential 모델로 시각 모델을 정의하자.
# 이 모델이 이미지를 벡터로 인코딩 하게 된다.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# 이제 시각 모델의 출력으로 텐서를 얻자:
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# 다음으로 질문을 벡터로 인코딩 하는 언어 모델을 정의하자.
# 각 질문은 최대 100 단어 길이이고
# 1에서 9999까지의 정수로 단어에 번호를 붙인다.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# 질문 벡터와 이미지 벡터를 이어 붙이자:
merged = keras.layers.concatenate([encoded_question, encoded_image])

# 1000 단어에 대해 로지스틱 회귀를 훈련시키자:
output = Dense(1000, activation='softmax')(merged)

# 우리의 최종 모델:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# 다음 단계는 실제 데이터로 이 모델을 훈련시키는 게 될 것이다.
```

### 영상 질의 응답 모델

이미지 QA 모델을 훈련해 봤으니 영상 QA 모델로도 금방 넘어갈 수 있다. 적절히 훈련시키면 짧은 비디오(가령 100 프레임짜리 사람의 동작)를 보여 주고서 그 영상에 대한 자연 언어 질문(가령 "소년이 하고 있는 운동 종목은?" -> "축구")을 할 수 있다.

```python
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# 앞서 훈련시킨 vision_model로 영상 인코딩 (가중치 재사용)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # 출력은 벡터들의 열
encoded_video = LSTM(256)(encoded_frame_sequence)  # 출력은 벡터 하나

# 모델 수준으로 표현된 질문 인코더. 앞서와 같은 가중치 재사용.
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# 질문을 인코딩 하자:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# 우리의 영상 질의 응답 모델:
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```
