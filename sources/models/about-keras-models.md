# 케라스 모델에 대해

케라스에서 사용 가능한 모델은 두 종류이다. [선형 모델](/models/sequential)과 [함수형 API로 사용하는 Model 클래스](/models/model)이다.

그 모델들에는 공통 메소드 및 속성이 몇 가지 있다.

- `model.layers`는 모델을 이루는 층들을 한 줄로 늘어놓은 리스트다.
- `model.inputs`는 모델의 입력 텐서들의 리스트다.
- `model.outputs`는 모델의 출력 텐서들의 리스트다.
- `model.summary()`는 모델 요약 설명을 찍는다. [utils.print_summary](/utils/#print_summary)의 바로가기.
- `model.get_config()`는 모델 설정을 담은 딕셔너리를 반환한다. 다음처럼 설정을 가지고 모델을 또 만들 수 있다.

```python
config = model.get_config()
model = Model.from_config(config)
# Sequential인 경우:
model = Sequential.from_config(config)
```

- `model.get_weights()`는 모델의 모든 가중치 텐서들의 리스트를 Numpy 배열들로 반환한다.
- `model.set_weights(weights)`는 Numpy 배열들의 리스트를 가지고 모델의 가중치 값들을 설정한다. 리스트의 배열들이 `get_weights()`에서 반환하는 것과 같은 형태여야 한다.
- `model.to_json()`은 모델의 JSON 문자열 표현을 반환한다. 참고로 그 표현에는 구조만 있고 가중치는 포함돼 있지 않다. 다음처럼 그 JSON 문자열을 가지고 동일한 (가중치들은 다시 초기화 된) 모델을 또 만들 수 있다.

```python
from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```
- `model.to_yaml()`은 모델의 YAML 문자열 표현을 반환한다. 참고로 그 표현에는 구조만 있고 가중치는 포함돼 있지 않다. 다음처럼 그 YAML 문자열을 가지고 동일한 (가중치들은 다시 초기화 된) 모델을 또 만들 수 있다.

```python
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)`는 모델의 가중치를 HDF5 파일로 저장한다.
- `model.load_weights(filepath, by_name=False)`는 (`save_weights()`로 만든) HDF5 파일에서 모델의 가중치를 불러온다. 기본적으로 구조가 바뀌어 있지 않아야 한다. 다른 (일부 층이 공통인) 구조로 가중치를 불러오려면 `by_name=True`를 써서 이름이 같은 층들만 불러오면 된다.

참고: `h5py` 설치 방법에 대해선 FAQ의 [케라스에서 모델을 저장하기 위해 HDF5 내지 h5py를 설치하려면?](/getting-started/faq/#hdf5-h5py) 항목을 보라.


## 모델 서브클래스 만들기

그 두 가지 모델 외에 원하는 대로 바꿀 수 있는 자체 모델을 만들 수도 있다.
`Model`의 서브클래스를 만들고 `call` 메소드에서 자체 진행 과정을 구현하면 된다.
(`Model` 서브클래스 생성 API는 케라스 2.2.0에서 도입됐다.)

다음은 `Model` 서브클래스로 작성된 간단한 다층 퍼셉트론 모델이다.

```python
import keras

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
```

`__init__(self, ...)`에서 층들을 정의하고 `call(self, inputs)`에서 진행 과정을 지정한다. `call`에서 `self.add_loss(loss_tensors)`를 호출해 (따로 만든 층에 하듯) 자체 제작한 손실을 지정할 수도 있다.

서브클래스 모델에서는 모델 구조를 (층들의 정적인 그래프가 아니라) 파이썬 코드로 정의한다.
따라서 모델 구조를 조사하거나 직렬화 할 수가 없다. 그래서 서브클래스 모델에서는 다음 메소드와 속성을 **사용할 수 없다**.

- `model.inputs` 및 `model.outputs`.
- `model.to_yaml()` 및 `model.to_json()`
- `model.get_config()` 및 `model.save()`.

**핵심 포인트:** 할 일에 맞는 API를 사용하라. `Model` 서브클래스 API를 쓰면 복잡한 모델 구현에 필요한 더 큰 유연성을 얻을 수 있기는 하지만
거기에는 (못 쓰게 되는 기능 말고도) 비용이 따른다.
더 길고 더 복잡하며 사용자 오류가 생길 여지가 더 많아진다. 가능하면 사용자 친화적인 함수형 API를 쓰는 게 좋다.
