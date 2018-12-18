# 응용 망

케라스 응용 망들은 미리 훈련된 가중치가 딸려 있어서 바로 사용 가능한 심층학습 모델들이다.
이 모델들을 예측, 피쳐 추출, 미세 조정 등에 사용할 수 있다.

모델 인스턴스 생성 시에 가중치를 자동으로 내려받는다. 그 가중치들은 `~/.keras/models/`에 저장된다.

## 쓸 수 있는 모델들

### ImageNet에서 가중치를 훈련한 이미지 분류 모델들:

- [Xception](#xception)
- [VGG16](#vgg16)
- [VGG19](#vgg19)
- [ResNet50](#resnet50)
- [InceptionV3](#inceptionv3)
- [InceptionResNetV2](#inceptionresnetv2)
- [MobileNet](#mobilenet)
- [DenseNet](#densenet)
- [NASNet](#nasnet)
- [MobileNetV2](#mobilenetv2)

이 구조들 전체가 모든 백엔드(텐서플로우, 테아노, CNTK)와 호환되며 인스턴스 생성 시 케라스 설정 파일 `~/.keras/keras.json`에 설정된 이미지 데이터 형식에 따라 모델이 구성된다. 예를 들어 `image_data_format=channels_last`라고 설정했다면 이 저장소로부터 적재하는 모델이 모두 텐서플로우의 주된 데이터 형식인 "높이-폭-깊이" 방식에 따라 구성된다.

참고 사항:

- `Keras < 2.2.0`에서, Xception 모델은 `SeparableConvolution` 층을 필요로 하기에 텐서플로우에만 사용 가능하다.
- `Keras < 2.1.5`에서, MobileNet 모델은 `DepthwiseConvolution` 층을 필요로 하기에 텐서플로우에만 사용 가능하다.

-----

## 이미지 분류 모델 사용 예

### ResNet50으로 ImageNet 클래스 분류하기

```python
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# 결과를 튜플 (class, description, probability)의 리스트로 디코딩
# (배치의 표본마다 그런 리스트가 하나씩 나옴)
print('Predicted:', decode_predictions(preds, top=3)[0])
# 예측: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
```

### VGG16으로 피쳐 추출하기

```python
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### VGG19로 임의의 중간 층에서 피쳐 추출하기

```python
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

### 새로운 클래스 집합에 대해 InceptionV3 미세 조정하기

```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# 기반이 되는 사전 훈련 모델 생성
base_model = InceptionV3(weights='imagenet', include_top=False)

# 전역 공간 평균 풀링 층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 완전 연결 층 추가
x = Dense(1024, activation='relu')(x)
# 로지스틱 층 -- 200가지 유형이 있다고 하자.
predictions = Dense(200, activation='softmax')(x)

# 이게 훈련시키게 될 모델
model = Model(inputs=base_model.input, outputs=predictions)

# 처음에: (난수로 초기화 한) 위쪽 층들만 훈련시킨다.
# 즉 합성곱인 InceptionV3 층들을 모두 얼려 둔다.
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일 (층들을 훈련 불가능하게 설정한 *후에* 해야 함)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 새 데이터로 몇 에포크 동안 모델 훈련
model.fit_generator(...)

# 이 시점에서는 위쪽 층들이 잘 훈련되었으므로 Inception V3의 합성곱 층
# 미세 조정을 시작할 수 있다. 아래의 N개 층을 얼려 두고 그 위의 나머지
# 층들을 훈련시키게 된다.

# 층을 몇 개나 얼려야 하는지 보기 위해 층 이름과 층 번호를 표시해 보기
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# Inception 블록들 중에 가장 위 2개를 순련시키기로 함. 즉 처음 249개
# 층을 얼리고 나머지를 풀어 줌.
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# 이 변경 내용들이 효과가 있으려면 모델을 다시 컴파일 해야 함
# 낮은 학습률로 SGD 사용
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# 모델 다시 훈련 (이번에는 위쪽 Dense 층들과 더불어 Inception 블록들 중
# 가장 위 2개를 함께 미세 조정)
model.fit_generator(...)
```


### 커스텀 입력 텐서 위에 InceptionV3 구성하기

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# 다른 케라스 모델이나 층의 출력일 수도 있음
input_tensor = Input(shape=(224, 224, 3))  # K.image_data_format() == 'channels_last' 가정

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```

-----

# 개별 모델 설명

| 모델 | 크기 | Top-1 정확도 | Top-5 정확도 | 매개변수 | 깊이 |
| ----- | ----: | --------------: | --------------: | ----------: | -----: |
| [Xception](#xception) | 88 MB | 0.790 | 0.945 | 22,910,480 | 126 |
| [VGG16](#vgg16) | 528 MB | 0.713 | 0.901 | 138,357,544 | 23 |
| [VGG19](#vgg19) | 549 MB | 0.713 | 0.900 | 143,667,240 | 26 |
| [ResNet50](#resnet50) | 99 MB | 0.749 | 0.921 | 25,636,712 | 168 |
| [InceptionV3](#inceptionv3) | 92 MB | 0.779 | 0.937 | 23,851,784 | 159 |
| [InceptionResNetV2](#inceptionresnetv2) | 215 MB | 0.803 | 0.953 | 55,873,736 | 572 |
| [MobileNet](#mobilenet) | 16 MB | 0.704 | 0.895 | 4,253,864 | 88 |
| [MobileNetV2](#mobilenetv2) | 14 MB | 0.713 | 0.901 | 3,538,984 | 88 |
| [DenseNet121](#densenet) | 33 MB | 0.750 | 0.923 | 8,062,504 | 121 |
| [DenseNet169](#densenet) | 57 MB | 0.762 | 0.932 | 14,307,880 | 169 |
| [DenseNet201](#densenet) | 80 MB | 0.773 | 0.936 | 20,242,984 | 201 |
| [NASNetMobile](#nasnet) | 23 MB | 0.744 | 0.919 | 5,326,716 | - |
| [NASNetLarge](#nasnet) | 343 MB | 0.825 | 0.960 | 88,949,818 | - |

Top-1 및 Top-5 정확도는 ImageNet 검증 데이터셋에 대한 모델 성능을 가리킨다.

-----


## Xception


```python
keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

Xception V1 모델. ImageNet으로 가중치 사전 훈련.

ImageNet에서 이 모델은 top-1 검증 정확도가 0.790,
top-5 검증 정확도가 0.945이다.

이 모델은 데이터 형식 `'channels_last'`(높이, 너비, 채널)만 지원한다.

이 모델의 기본 입력 크기는 299x299이다.

### 인자

- include_top: 망 가장 위에 완전 연결 층을 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는 `'imagenet'`(ImageNet으로 사전 훈련).
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸 케라스 텐서(즉 `layers.Input()`의 출력).
- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    `(299, 299, 3)`이어야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 71보다 작지 않아야 함.
    가령 `(150, 150, 3)`이 유효한 값.
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 `Model` 인스턴스.

### 참고 자료

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### 라이선스

이 가중치는 우리가 훈련시킨 것이며 MIT 라이선스에 따라 공개되어 있다.


-----


## VGG16

```python
keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

VGG16 모델. ImageNet으로 가중치 사전 훈련.

이 모델은 `'channels_first'` 데이터 형식(채널, 높이, 너비)이나 `'channels_last'` 데이터 형식(높이, 너비, 채널) 어느 쪽으로도 구성할 수 있다.

이 모델의 기본 입력 크기는 224x224이다.

### 인자

- include_top: 망 가장 위에 완전 연결 층 3개를 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는 `'imagenet'`(ImageNet으로 사전 훈련).
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸 케라스 텐서(즉 `layers.Input()`의 출력).
- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    `(224, 224, 3)`(`'channels_last'` 데이터 형식)이나
    `(3, 224, 224)`(`'channels_first'` 데이터 형식)여야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 32보다 작지 않아야 함.
    가령 `(200, 200, 3)`이 유효한 값.
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 `Model` 인스턴스.

### 참고 자료

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556): 작업물에 VGG 모델을 쓰는 경우 부디 이 논문을 인용해 달라.

### 라이선스

이 가중치는 [옥스포드의 VGG에서](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [Creative Commons 저작자 표시 라이선스](https://creativecommons.org/licenses/by/4.0/)로 공개한 것을 이식한 것이다.

-----

## VGG19


```python
keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```


VGG19 모델. ImageNet으로 가중치 사전 훈련.

이 모델은 `'channels_first'` 데이터 형식(채널, 높이, 너비)이나 `'channels_last'` 데이터 형식(높이, 너비, 채널) 어느 쪽으로도 구성할 수 있다.

이 모델의 기본 입력 크기는 224x224이다.

### 인자

- include_top: 망 가장 위에 완전 연결 층 3개를 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는 `'imagenet'`(ImageNet으로 사전 훈련).
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸 케라스 텐서(즉 `layers.Input()`의 출력).
- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    `(224, 224, 3)`(`'channels_last'` 데이터 형식)이나
    `(3, 224, 224)`(`'channels_first'` 데이터 형식)여야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 32보다 작지 않아야 함.
    가령 `(200, 200, 3)`이 유효한 값.
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 `Model` 인스턴스.

### 참고 자료

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### 라이선스

이 가중치는 [옥스포드의 VGG에서](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [Creative Commons 저작자 표시 라이선스](https://creativecommons.org/licenses/by/4.0/)로 공개한 것을 이식한 것이다.

-----

## ResNet50


```python
keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```


ResNet50 모델. ImageNet으로 가중치 사전 훈련.

이 모델은 `'channels_first'` 데이터 형식(채널, 높이, 너비)이나 `'channels_last'` 데이터 형식(높이, 너비, 채널) 어느 쪽으로도 구성할 수 있다.

이 모델의 기본 입력 크기는 224x224이다.


### 인자

- include_top: 망 가장 위에 완전 연결 층을 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는 `'imagenet'`(ImageNet으로 사전 훈련).
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸 케라스 텐서(즉 `layers.Input()`의 출력).
- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    `(224, 224, 3)`(`'channels_last'` 데이터 형식)이나
    `(3, 224, 224)`(`'channels_first'` 데이터 형식)여야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 32보다 작지 않아야 함.
    가령 `(200, 200, 3)`이 유효한 값.
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 `Model` 인스턴스.

### 참고 자료

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### 라이선스

이 가중치는 [Kaiming He가](https://github.com/KaimingHe/deep-residual-networks) [MIT 라이선스](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE)로 공개한 것을 이식한 것이다.

-----

## InceptionV3


```python
keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

Inception V3 모델. ImageNet으로 가중치 사전 훈련.

이 모델은 `'channels_first'` 데이터 형식(채널, 높이, 너비)이나 `'channels_last'` 데이터 형식(높이, 너비, 채널) 어느 쪽으로도 구성할 수 있다.

이 모델의 기본 입력 크기는 299x299이다.


### 인자

- include_top: 망 가장 위에 완전 연결 층을 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는 `'imagenet'`(ImageNet으로 사전 훈련).
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸 케라스 텐서(즉 `layers.Input()`의 출력).
- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    `(299, 299, 3)`(`'channels_last'` 데이터 형식)이나
    `(3, 299, 299)`(`'channels_first'` 데이터 형식)여야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 75보다 작지 않아야 함.
    가령 `(150, 150, 3)`이 유효한 값.
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 `Model` 인스턴스.

### 참고 자료

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

### 라이선스

이 가중치는 [아파치 라이선스](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 공개되어 있다.

-----

## InceptionResNetV2


```python
keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

Inception-ResNet V2 모델. ImageNet으로 가중치 사전 훈련.

이 모델은 `'channels_first'` 데이터 형식(채널, 높이, 너비)이나 `'channels_last'` 데이터 형식(높이, 너비, 채널) 어느 쪽으로도 구성할 수 있다.

이 모델의 기본 입력 크기는 299x299이다.


### 인자

- include_top: 망 가장 위에 완전 연결 층을 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는 `'imagenet'`(ImageNet으로 사전 훈련).
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸 케라스 텐서(즉 `layers.Input()`의 출력).
- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    `(299, 299, 3)`(`'channels_last'` 데이터 형식)이나
    `(3, 299, 299)`(`'channels_first'` 데이터 형식)여야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 75보다 작지 않아야 함.
    가령 `(150, 150, 3)`이 유효한 값.
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 `Model` 인스턴스.

### 참고 자료

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

### 라이선스

이 가중치는 [아파치 라이선스](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 공개되어 있다.

-----

## MobileNet


```python
keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

MobileNet 모델. ImageNet으로 가중치 사전 훈련.

이 모델은 데이터 형식 `'channels_last'`(높이, 너비, 채널)만 지원한다.

이 모델의 기본 입력 크기는 224x224이다.

### 인자

- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    `(224, 224, 3)`(`'channels_last'` 데이터 형식)이나
    `(3, 224, 224)`(`'channels_first'` 데이터 형식)여야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 32보다 작지 않아야 함.
    가령 `(200, 200, 3)`이 유효한 값.
- alpha: 망의 너비를 제어함.
    - `alpha` < 1.0이면 각 층에서 필터 수를
        비례해서 줄임.
    - `alpha` > 1.0이면 각 층에서 필터 수를
        비례해서 늘임.
    - `alpha` = 1이면 각 층에 논문에 있는
        기본 필터 수를 사용함.
- depth_multiplier: 깊이 방향 합성곱의 깊이 승수.
    (해상도 승수라고도 함.)
- include_top: 망 가장 위에 완전 연결 층을 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는 `'imagenet'`(ImageNet 가중치).
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸
    케라스 텐서(즉 `layers.Input()`의 출력).
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 `Model` 인스턴스.

### 참고 자료

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

### 라이선스

이 가중치는 [아파치 라이선스](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 공개되어 있다.

-----

## DenseNet


```python
keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

DenseNet 모델. ImageNet으로 가중치 사전 훈련.

이 모델은 `'channels_first'` 데이터 형식(채널, 높이, 너비)이나 `'channels_last'` 데이터 형식(높이, 너비, 채널) 어느 쪽으로도 구성할 수 있다.

이 모델의 기본 입력 크기는 224x224이다.

### 인자

- blocks: 4개 밀집 층에서의 구성 블록 수.
- include_top: 망 가장 위에 완전 연결 층을 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는
    'imagenet'(ImageNet으로 사전 훈련) 또는
    적재할 가중치 파일의 경로.
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸
    케라스 텐서(즉 `layers.Input()`의 출력).
- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    `(224, 224, 3)`(`'channels_last'` 데이터 형식)이나
    `(3, 224, 224)`(`'channels_first'` 데이터 형식)여야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 32보다 작지 않아야 함.
    가령 `(200, 200, 3)`이 유효한 값.
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 모델 인스턴스.

### 참고 자료

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

### 라이선스

이 가중치는 [BSD 3조항 라이선스](https://github.com/liuzhuang13/DenseNet/blob/master/LICENSE)에 따라 공개되어 있다.

-----

## NASNet


```python
keras.applications.nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

Neural Architecture Search Network (NASNet) 모델. ImageNet으로 가중치 사전 훈련.

NASNetLarge 모델의 기본 입력 크기는 331x331이고
NASNetMobile 모델은 224x224이다.

### 인자

- input_shape: 선택적. 형태 튜플이며 `include_top`이
    `False`인 경우에만 지정. 아닌 경우에는 입력 형태가
    NASNetMobile에서는 `(224, 224, 3)`(`'channels_last'`
    데이터 형식)이나 `(3, 224, 224)`(`'channels_first'`
    데이터 형식)여야 하고 NASNetLarge에서는
    `(331, 331, 3)`(`'channels_last'` 데이터 형식)이나
    `(3, 331, 331)`(`'channels_first'` 데이터 형식)이어야 함.
    입력 채널이 정확히 3개여야 하고
    너비와 높이가 32보다 작지 않아야 함.
    가령 `(200, 200, 3)`이 유효한 값.
- include_top: 망 가장 위에 완전 연결 층을 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는 `'imagenet'`(ImageNet 가중치).
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸
    케라스 텐서(즉 `layers.Input()`의 출력).
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 `Model` 인스턴스.

### 참고 자료

- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

### 라이선스

이 가중치는 [아파치 라이선스](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 공개되어 있다.

-----

## MobileNetV2


```python
keras.applications.mobilenetv2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

MobileNetV2 모델. ImageNet으로 가중치 사전 훈련.

이 모델은 데이터 형식 `'channels_last'`(높이, 너비, 채널)만 지원한다.

이 모델의 기본 입력 크기는 224x224이다.

### 인자

- input_shape: 선택적. 형태 튜플이며 해상도가 (224, 224, 3)이
    아닌 입력 이미지에 모델을 쓰고 싶은 경우에 지정.
    입력 채널이 정확히 3개여야 함 (224, 224, 3).
    input_tensor에서 input_shape을 추론하게 하고
    싶은 경우에도 이 옵션을 생략할 수 있다.
    input_tensor와 input_shape를 모두 주기로 한 경우에는
    둘이 일치하면 input_shape을 사용하고
    형태가 일치하지 않으면 오류를 던진다.
    가령 `(160, 160, 3)`이 유효한 값.
- alpha: 망의 너비를 제어함. MobileNetV2 논문에서는
    너비 승수라고 함.
    - `alpha` < 1.0이면 각 층에서 필터 수를
        비례해서 줄임.
    - `alpha` > 1.0이면 각 층에서 필터 수를
        비례해서 늘임.
    - `alpha` = 1이면 각 층에 논문에 있는
        기본 필터 수를 사용함.
- depth_multiplier: 깊이 방향 합성곱의 깊이 승수.
      (해상도 승수라고도 함.)
- include_top: 망 가장 위에 완전 연결 층을 포함시킬지 여부.
- weights: `None`(난수 초기화) 또는
        'imagenet'(ImageNet으로 사전 훈련) 또는
        적재할 가중치 파일의 경로.
- input_tensor: 선택적. 모델의 이미지 입력으로 쓸
      케라스 텐서(즉 `layers.Input()`의 출력).
- pooling: 선택적. `include_top`이 `False`일 때
    피쳐 추출을 위한 풀링 방식.
    - `None`은 마지막 합성곱 층의 4차원 텐서
        출력이 모델의 출력이 된다는 뜻이다.
    - `'avg'`는 마지막 합성곱 층의 출력에
        전역 평균 풀링을 적용한다는 뜻이며,
        그래서 모델의 출력이 2차원 텐서가 된다.
    - `'max'`는 전역 맥스 풀링을 적용한다는 뜻이다.
- classes: 선택적. 이미지를 분류할 유형의 수이며
    `include_top`이 `True`이고 `weights` 인자를
    지정하지 않은 경우에만 지정하면 됨.

### 반환

케라스 모델 인스턴스.

### 예외

ValueError: `weights` 인자가 유효하지 않거나,
    weights='imagenet'일 때 입력 형태가 유효하지 않거나
    depth_multiplier, alpha, rows가 유효하지 않은 경우.

### 참고 자료

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

### 라이선스

이 가중치는 [아파치 라이선스](https://github.com/tensorflow/models/blob/master/LICENSE)에 따라 공개되어 있다.
