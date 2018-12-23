# 새로운 케라스 층 작성

단순하고 상태 없는 연산이라면 아마 `layers.core.Lambda` 층을 쓰는 게 나을 것이다. 하지만 훈련 가능한 가중치가 있는 연산이라면 새로 층을 구현해야 한다.

다음은 **케라스 2.0을 기준으로 한** (이전 버전을 쓰고 있다면 업그레이드 하자) 케라스 층의 골격이다. 메소드 세 개만 구현하면 된다.

- `build(input_shape)`: 가중치를 여기서 정의하게 된다. 이 메소드 마지막에서 `self.built = True` 설정을 해야 하는데, `super([Layer], self).build()`를 호출하는 것으로도 가능하다.
- `call(x)`: 층의 로직이 있는 곳이다. 층에서 마스킹을 지원하게 하고 싶은 게 아니라면 `call`의 첫 번째 인자인 입력 텐서에만 신경 쓰면 된다.
- `compute_output_shape(input_shape)`: 층에서 입력의 형태를 변경하는 경우에는 여기서 형태 변환 로직을 명시해야 한다. 이를 통해 케라스가 자동 형태 추론을 할 수 있게 된다.

```python
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 이 층에서 쓸 훈련 가능한 가중치 변수 만들기
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 마지막에 이걸 꼭 호출해야 함

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

입력 텐서와 출력 텐서가 여럿인 케라스 층을 정의하는 것도 가능하다. 그러자면 메소드 `build(input_shape)`, `call(x)`, `compute_output_shape(input_shape)`의 입력과 출력이 리스트라고 상정해야 한다. 위와 비슷한 예는 다음과 같다.

```python
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 이 층에서 쓸 훈련 가능한 가중치 변수 만들기
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 마지막에 이걸 꼭 호출해야 함

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]
```

이미 있는 케라스 층들은 거의 모든 걸 구현하는 방법에 대한 예시가 돼 준다. 소스 코드 읽는 걸 망설이지 말자!
