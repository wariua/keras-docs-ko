# 케라스 층에 대해

모든 케라스 층에는 공통 메소드가 몇 가지 있다.

- `layer.get_weights()`: 층의 가중치들을 Numpy 배열들의 리스트로 반환.
- `layer.set_weights(weights)`: (`get_weights`의 출력과 형태가 같은) Numpy 배열들의 리스트를 가지고 층의 가중치 설정.
- `layer.get_config()`: 층의 설정을 담은 딕셔너리 반환. 다음처럼 설정을 가지고 층의 인스턴스를 또 만들 수 있다.

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

또는 다음처럼:

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
```

층에 노드가 하나이면 (즉 공유 층이 아니면) 다음을 통해 입력 텐서, 출력 텐서, 입력 형태, 출력 형태를 얻을 수 있다.

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

층에 노드가 여러 개이면 ([층 노드와 공유 층 개념](/getting-started/functional-api-guide/#_5) 참고) 다음 메소드를 쓸 수 있다.

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`
