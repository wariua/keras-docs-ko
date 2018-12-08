<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/embeddings.py#L15)</span>
### Embedding

```python
keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

양의 정수들(인덱스)을 고정 크기의 밀집 벡터로 바꾼다.
예: [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]].

이 층은 모델의 첫 번째 층으로만 쓸 수 있다.

__예시__


```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# 크기가 (batch, input_length)인 정수 행렬을 모델이 입력으로 받게 된다.
# 입력에서 가장 큰 정수(즉 단어 인덱스)가 999(어휘 수)보다 커서는 안 된다.
# 그러면 model.output_shape = (None, 10, 64)이다. None은 배치 차원이다.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

__인자__

- __input_dim__: int > 0. 어휘 수.
    즉 가장 큰 정수 인덱스 + 1.
- __output_dim__: int >= 0. 밀집 embedding의 차원.
- __embeddings_initializer__: `embeddings` 행렬 초기화 방식.
    ([초기화](../initializers.md) 참고.)
- __embeddings_regularizer__: `embeddings` 행렬에 적용하는
    정칙화 함수.
    ([정칙화](../regularizers.md) 참고.)
- __embeddings_constraint__: `embeddings` 행렬에 적용하는
    제약 함수.
    ([제약](../constraints.md) 참고.)
- __mask_zero__: 입력 값 0이 감춰 버려야 하는
    "패딩" 특수 값인지 여부.
    가변 길이 입력을 받을 수도 있는
    [순환 층](recurrent.md)을 쓸 때 유용하다.
    이 값이 `True`이면 모델의 후속 층 모두에서
    마스킹을 지원해야 하며, 안 그러면 예외가 발생한다.
    mask_zero가 True로 설정돼 있으면 그로 인해
    어휘에서 인덱스 0을 쓸 수 없다. (input_dim이
    어휘 수 + 1이어야 한다.)
- __input_length__: 입력 열의 길이가 상수인 경우 그 길이.
    위로 `Flatten` 층과 `Dense` 층을 차례로 연결하려면
    이 인자가 꼭 있어야 한다.
    (이 인자 없이는 밀집 출력의 형태를 계산할 수 없다.)

__입력 형태__

`(batch_size, sequence_length)` 형태의 2차원 텐서.

__출력 형태__

`(batch_size, sequence_length, output_dim)` 형태의 3차원 텐서.

__참고 자료__

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
