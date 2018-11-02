# 케라스: 파이썬 심층학습 라이브러리

<img src='https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png', style='max-width: 600px;'>



## 케라스와의 첫 만남

케라스(Keras)는 파이썬으로 작성된 고수준 신경망 API이며 [텐서플로우](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), [테아노](https://github.com/Theano/Theano) 위에서 동작할 수 있다. 빠른 실험을 가능하게 하는 데 중점을 두고 개발되었다. *최소한의 지연으로 아이디어에서 결과까지 가는 건 좋은 연구를 위한 열쇠이다.*

다음과 같은 심층학습 라이브러리가 필요하다면 케라스를 쓰면 된다.

- 쉽고 빠르게 프로토타입 제작이 가능 (사용자 친화성, 모듈성, 확장성)
- 합성곱 망과 순환 망, 그리고 둘의 조합까지 지원
- CPU와 GPU 위에서 매끄럽게 동작

[Keras.io](https://keras.io)에서 문서를 볼 수 있다.

케라스는 __파이썬 2.7-3.6__과 호환된다.


------------------


## 주요 원칙

- __사용자 친화성.__ 케라스는 기계가 아니라 사람을 위해 설계된 API이다. 사용자 경험을 전면에, 그리고 가운데에 둔다. 케라스는 인지 부하를 줄이기 위한 검증된 관행들을 따른다. 즉 일관성 있고 단순한 API를 제공하고, 흔히 쓰는 방식에서 필요한 사용자 동작을 가급적 줄이며, 사용자 오류에 대해 분명하고 대처 가능한 피드백을 제공한다.

- __모듈성.__ 모델이란 거의 제약 없이 서로에게 끼워 맞출 수 있는 독립적이고 완전히 제어 가능한 열 내지 그래프라고 생각하면 된다. 특히 신경 층, 비용 함수, 옵티마이저, 초기화 체계, 활동 함수, 정규화 체계 모두가 독립적 모듈이고 이를 합쳐서 새 모델을 만들 수 있다.

- __쉬운 확장.__ 새 모듈을 간단하게 (새 클래스나 함수처럼) 추가할 수 있고 기존 모듈들의 풍부한 예시가 있다. 새 모듈을 쉽게 만들 수 있다는 것은 완전한 표현이 가능하다는 것이고, 그래서 선진적인 연구에 케라스가 적합하다.

- __파이썬으로 동작.__ 화려한 형식으로 된 별도의 모델 설정 파일이 없다. 파이썬 코드로 모델을 기술한다. 간결하고 디버깅이 쉬우며 손쉬운 확장이 가능하다.


------------------


## 써 보기: 30초에 보는 케라스

케라스의 핵심 자료 구조는 __모델__인데 이는 층(layer)들을 조직하는 방식이다. 가장 간단한 종류의 모델은 [`Sequential`](https://keras.io/getting-started/sequential-model-guide) 모델로, 층들을 차례로 쌓는다. 더 복잡한 구조를 위해선 [케라스 함수형 API](https://keras.io/getting-started/functional-api-guide)를 써야 하는데, 그러면 층들로 어떤 그래프든 만들 수 있다.

다음은 `Sequential` 모델이다.

```python
from keras.models import Sequential

model = Sequential()
```

층을 쌓으려면 `.add()`만 하면 된다.

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

모델이 괜찮다 싶으면 `.compile()`로 학습 과정을 구성하면 된다.

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요하다면 옵티마이저를 더 자세하게 설정할 수 있다. 케라스의 핵심 원칙은 작업을 충분히 단순하게 만들되 필요시 사용자가 완전하게 제어할 수 있도록 하는 것이다. (궁극적인 제어는 소스 코드를 손쉽게 확장할 수 있는 것이다.)
```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

이제 훈련 데이터로 배치 반복을 할 수 있다.

```python
# x_train과 y_train은 Numpy 배열 --Scikit-Learn API와 동일.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

아니면 수동으로 모델에 배치를 넣어 줄 수 있다.

```python
model.train_on_batch(x_batch, y_batch)
```

한 줄로 성능을 평가할 수 있다.

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

그리고 새 데이터에 대한 예측을 만들 수 있다.

```python
classes = model.predict(x_test, batch_size=128)
```

질의 응답 시스템, 이미지 분류 모델, 뉴럴 튜링 머신, 기타 다른 모델도 마찬가지로 빠르게 만들 수 있다. 심층학습의 기본 아이디어는 단순한데 그걸 구현하는 게 고통스러워서야 되겠는가?

더 깊이 들어가는 케라스 튜토리얼을 다음에서 볼 수 있다.

- [Sequential 모델 써 보기](https://keras.io/getting-started/sequential-model-guide)
- [함수형 API 써 보기](https://keras.io/getting-started/functional-api-guide)

저장소의 [예시 폴더](https://github.com/keras-team/keras/tree/master/examples)에서 더 복잡한 모델들을 찾을 수 있다. 메모리 망을 쓰는 질의 응답, LSTM을 중첩한 텍스트 생성 등이 있다.


------------------


## 설치

케라스를 설치하기 전에 백엔드 엔진으로 텐서플로우, 테아노, CNTK 중 하나를 설치해야 한다. 텐서플로우 백엔드를 권한다.

- [텐서플라우 설치 절차](https://www.tensorflow.org/install/).
- [테아노 설치 절차](http://deeplearning.net/software/theano/install.html#install).
- [CNTK 설치 절차](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine).

다음 **선택적 의존 프로그램** 설치를 고려해 볼 수도 있다.

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (케라스를 GPU에서 돌릴 계획이면 권장).
- HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (케라스 모델을 디스크에 저장할 계획이면 필수).
- [graphviz](https://graphviz.gitlab.io/download/) and [pydot](https://github.com/erocarrera/pydot) ([시각화 도구](https://keras.io/visualization/)에서 모델 그래프를 그리는 데 사용).

이제 케라스를 설치할 수 있다. 케라스를 설치하는 방법이 두 가지 있다.

- **PyPI로 케라스 설치하기 (권장):**

```sh
sudo pip install keras
```

virtualenv를 쓰고 있다면 sudo 사용을 피할 수도 있다.

```sh
pip install keras
```

- **아니면: 깃허브 소스로 케라스 설치하기:**

먼저 `git`으로 케라스를 복제한다.

```sh
git clone https://github.com/keras-team/keras.git
```

 그 다음 케라스 폴더로 `cd` 해서 설치 명령을 실행한다.
```sh
cd keras
sudo python setup.py install
```

------------------


## 케라스 백엔드 구성하기

기본적으로 케라스에서는 텐서 조작 라이브러리로 텐서플로우를 쓴다. 케라스 백엔드 구성을 위해선 [이 설명](https://keras.io/backend/)을 따르면 된다.

------------------


## 지원

다음에서 질문을 하거나 개발 논의에 참여할 수 있다.

- [케라스 구글 그룹](https://groups.google.com/forum/#!forum/keras-users).
- [케라스 슬랙 채널](https://kerasteam.slack.com). 채널 초대를 요청하려면 [이 링크](https://keras-slack-autojoin.herokuapp.com/)를 이용하면 된다.

또 [깃허브 이슈](https://github.com/keras-team/keras/issues)로(만) **버그 보고 및 기능 요청**을 올릴 수 있다. 먼저 [가이드라인](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)을 꼭 읽어 보라.


------------------


## 왜 이름이 케라스인가?

케라스(Keras, κέρας)는 그리스어로 *뿔*을 뜻한다. 그리스 및 라틴 문헌에서 온 문학적 이미지를 나타내는데 첫 등장은 *오디세이아*이다. 거기서 꿈의 정령들(_오네이로이_)은 상아의 문으로 지상에 와서 거짓 환영으로 사람을 속이는 쪽과 뿔의 문으로 와서 앞으로의 미래를 알려 주는 쪽으로 나뉜다. κέρας (뿔) / κραίνω (실현), ἐλέφας (상아) / ἐλεφαίρομαι (기만)으로 말장난하는 것이다.

처음에 케라스는 ONEIROS(Open-ended Neuro-Electronic Intelligent Robot Operating System) 프로젝트 연구 활동의 일부로 개발되었다.

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ 호메로스, 오디세이아 19. 562 ff (Shewring 번역).

------------------
