<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L212)</span>
### RNN

```python
keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

순환 층의 기반 클래스.

__인자__

- __cell__: RNN 셀 인스턴스. RNN 셀은 다음을 포함한 클래스이다.
    - `call(input_at_t, states_at_t)` 메소드.
        `(output_at_t, states_at_t_plus_1)`을 반환.
        셀의 call 메소드가 선택적으로 `constants` 인자를 받을 수도
        있다. 아래의 "외부 상수 전달에 대해" 절 참고.
    - `state_size` 속성. 정수 한 개(단일 상태)일 수 있으며,
        그 경우에는 순환 상태의 크기이다.
        (그 크기가 셀 출력의 크기와 같아야 한다.)
        또한 (상태별로 크기 하나씩) 정수 리스트/튜플일
        수도 있다. 그 경우에 첫 번째 항목(`state_size[0]`)이
        셀 출력의 크기와 같아야 한다.

    `cell`이 RNN 셀 인스턴스들의 리스트일 수도 있으며,
    그 경우에는 RNN 내에서 셀들이 쌓여서
    효율적인 다층 RNN을 구현한다.

- __return_sequences__: 불리언. 출력 열에 최종 출력을 반환할
    것인지 아니면 전체 열을 반환할 것인지.
- __return_state__: 불리언. 최종 상태를 출력에 더해서 반환할
    것인지 여부.
- __go_backwards__: 불리언 (기본은 False).
    True이면 입력 열을 역방향으로 처리하고
    뒤집힌 열을 반환한다.
- __stateful__: 불리언 (기본은 False). True이면
    배치에서 i 번째 표본의 마지막 상태가 각각 이어지는
    배치에서 i 번째 표본의 초기 상태로 쓰이게 된다.
- __unroll__: 불리언 (기본은 False).
    True이면 망을 전개하고
    아니면 symbolic 루프를 쓴다.
    전개를 하면 RNN의 속도를 높일 수 있지만
    메모리를 더 소모하게 된다.
    짧은 열에만 전개가 적합하다.
- __input_dim__: 입력의 차원(정수).
    이 층을 모델의 첫 번째 층으로 사용할 때
    이 인자가 (아니면 키워드 인자 `input_shape`가)
    꼭 있어야 한다.
- __input_length__:  입력 열의 길이로,
    상수일 때 지정하면 됨.
    위쪽으로 `Flatten` 층에 이어 `Dense` 층을
    연결하려는 경우 이 인자가 꼭 필요함.
    (이 인자 없이는 밀집 출력의 형태를 계산할 수 없음.)
    참고로 순환 층이 모델의 첫 번째 층이 아니라면
    첫째 층 단계에서 입력 길이를
    (가령 `input_shape` 인자를 통해)
    지정해 주어야 할 것이다.

__입력 형태__

`(batch_size, timesteps, input_dim)` 형태의 3차원 텐서.

__출력 형태__

- `return_state`이면: 텐서들의 리스트. 첫 번째 텐서가
    출력이다. 나머지 텐서들은 각각 `(batch_size, units)`
    형태인 최종 상태이다.
- `return_sequences`이면: `(batch_size, timesteps, units)`
    형태의 3차원 텐서.
- 아니면, `(batch_size, units)` 형태의 2차원 텐서.

__마스킹__

이 층은 가변 개수 timestep의 입력 데이터에 대한
마스킹을 지원한다. 데이터에 마스크를 적용하려면
[Embedding](embeddings.md) 층을 `mask_zero` 매개변수를
`True`로 설정해서 사용하면 된다.

__RNN에서 상태 유지 방식 사용하기__

RNN 층을 '상태 유지 방식'으로 설정할 수 있다. 그러면 어떤 배치의
표본들에 대해 계산한 상태들을 다음 배치의 표본들에 대한
초기 상태로 재사용하게 된다. 이 방식은 연속한 배치들의
표본들 간에 일대일 관계가 있다고 상정한다.

상태 유지 방식을 켜려면

- 층 생성자에 `stateful=True`를 지정한다.
- 모델에 고정 배치 크기를 지정한다.
순차 모델이라면 모델 첫 번째 층에
`batch_input_shape=(...)`를 주면 되고,
함수형 모델이라면 모델 첫 번째 층에
`batch_shape=(...)`를 주면 된다.
이는 *배치 크기를 포함하는*
예상 입력 형태이다.
정수들의 튜플이어야 한다. 가령 `(32, 10, 100)`.
- fit() 호출 시 `shuffle=False`를 지정한다.

모델의 상태를 초기화하려면 특정 층에 대해, 또는
모델 전체에 대해 `.reset_states()`를 호출하면 된다.

__RNN의 초기 상태 지정하기__

키워드 인자 `initial_state`로 RNN 층을 호출해서 그 층의
초기 상태를 심볼로 지정할 수 있다. `initial_state`의 값은
RNN 층의 초기 상태를 나타내는 텐서 내지 텐서들의 리스트여야 한다.

키워드 인자 `states`로 `reset_states`를 호출해서 RNN 층의
초기 상태를 수치로 지정할 수 있다. `states`의 값은
RNN 층의 초기 상태를 나타내는 numpy 배열 내지 
numpy 배열들의 리스트여야 한다.

__외부 상수를 RNN으로 전달하기__

`RNN.__call__` (및 `RNN.call`) 메소드의 키워드 인자 `constants`를
이용해 "외부" 상수들을 셀로 전달할 수 있다. 이를 위해선
`cell.call` 메소드가 동일 키워드 인자 `constants`를 받아야 한다.
그런 상수들을 이용해 추가적인 (시간에 따라 바뀌지 않는)
정적 입력으로 셀의 변형 방식에 영향을 줄 수 있다.
즉 주목(attention) 메커니즘이다.

__예시__


```python
# 먼저 층의 서브클래스로 RNN 셀을 정의하자.

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# RNN 층에 그 셀을 사용하자.

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# 셀을 다음처럼 사용해서 다층 RNN을 만들 수 있다.

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L902)</span>
### SimpleRNN

```python
keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

출력을 입력으로 되먹이는 완전 연결 RNN.

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 쌍곡탄젠트(`tanh`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __dropout__: 0과 1 사이의 float.
    입력의 선형 변환에 대해
    버릴 유닛들의 비율.
- __recurrent_dropout__: 0과 1 사이의 float.
    순환 상태의 선형 변환에 대해
    버릴 유닛들의 비율.
- __return_sequences__: 불리언. 출력 열에 최종 출력을 반환할
    것인지 아니면 전체 열을 반환할 것인지.
- __return_state__: 불리언. 최종 상태를 출력에 더해서 반환할
    것인지 여부.
- __go_backwards__: 불리언 (기본은 False).
    True이면 입력 열을 역방향으로 처리하고
    뒤집힌 열을 반환한다.
- __stateful__: 불리언 (기본은 False). True이면
    배치에서 i 번째 표본의 마지막 상태가 각각 이어지는
    배치에서 i 번째 표본의 초기 상태로 쓰이게 된다.
- __unroll__: 불리언 (기본은 False).
    True이면 망을 전개하고
    아니면 symbolic 루프를 쓴다.
    전개를 하면 RNN의 속도를 높일 수 있지만
    메모리를 더 소모하게 된다.
    짧은 열에만 전개가 적합하다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1426)</span>
### GRU

```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)
```

게이트 순환 유닛 - 조경현 외 2014.

두 가지 방식이 있다. 기본 방식은 1406.1078v3을 기반으로 한 것으로
행렬 곱셈 전에 은닉 상태에 리셋 게이트를 적용시킨다. 다른 방식은
원저 논문 1406.1078v1을 기반으로 한 것으로 순서가 반대이다.

두 번째 방식은 (GPU 전용인) CuDNNGRU와 호환되며 추론은 CPU에서도
가능하다. 그래서 `kernel`과 `recurrent_kernel`에 별도의 편향이 있다.
`'reset_after'=True` 및 `recurrent_activation='sigmoid'`라고
하면 된다.

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 쌍곡탄젠트(`tanh`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __recurrent_activation__: 순환 단계에
    사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 하드 시그모이드 (`hard_sigmoid`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __dropout__: 0과 1 사이의 float.
    입력의 선형 변환에 대해
    버릴 유닛들의 비율.
- __recurrent_dropout__: 0과 1 사이의 float.
    순환 상태의 선형 변환에 대해
    버릴 유닛들의 비율.
- __implementation__: 구현 모드. 1 또는 2.
    모드 1에서는 연산들을 많은 수의
    작은 도트곱과 덧셈들로 조직하는 반면
    모드 2에서는 적은 수의 큰 연산들로 묶는다.
    두 모드는 하드웨어와 응용에 따라
    상이한 성능 특성을 보이게 된다.
- __return_sequences__: 불리언. 출력 열에 최종 출력을 반환할
    것인지 아니면 전체 열을 반환할 것인지.
- __return_state__: 불리언. 최종 상태를 출력에 더해서 반환할
    것인지 여부.
- __go_backwards__: 불리언 (기본은 False).
    True이면 입력 열을 역방향으로 처리하고
    뒤집힌 열을 반환한다.
- __stateful__: 불리언 (기본은 False). True이면
    배치에서 i 번째 표본의 마지막 상태가 각각 이어지는
    배치에서 i 번째 표본의 초기 상태로 쓰이게 된다.
- __unroll__: 불리언 (기본은 False).
    True이면 망을 전개하고
    아니면 symbolic 루프를 쓴다.
    전개를 하면 RNN의 속도를 높일 수 있지만
    메모리를 더 소모하게 된다.
    짧은 열에만 전개가 적합하다.
- __reset_after__: GRU 동작 방식 (리셋 게이트를 행렬 곱셈 후에
    적용할지 전에 적용할지). False = "전에 적용" (기본값),
    True = "후에 적용" (CuDNN 호환).

__참고 자료__

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1957)</span>
### LSTM

```python
keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

장단기 기억 층 - Hochreiter 1997.

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 쌍곡탄젠트(`tanh`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __recurrent_activation__: 순환 단계에
    사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 하드 시그모이드 (`hard_sigmoid`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __unit_forget_bias__: 불리언.
    True이면 초기화 시 망각 게이트 편향에 1을 더한다.
    참으로 설정하면 `bias_initializer="zeros"`도 써야 한다.
    [Jozefowicz 외의 논문](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)에서 이를 권장한다.
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __dropout__: 0과 1 사이의 float.
    입력의 선형 변환에 대해
    버릴 유닛들의 비율.
- __recurrent_dropout__: 0과 1 사이의 float.
    순환 상태의 선형 변환에 대해
    버릴 유닛들의 비율.
- __implementation__: 구현 모드. 1 또는 2.
    모드 1에서는 연산들을 많은 수의
    작은 도트곱과 덧셈들로 조직하는 반면
    모드 2에서는 적은 수의 큰 연산들로 묶는다.
    두 모드는 하드웨어와 응용에 따라
    상이한 성능 특성을 보이게 된다.
- __return_sequences__: 불리언. 출력 열에 최종 출력을 반환할
    것인지 아니면 전체 열을 반환할 것인지.
- __return_state__: 불리언. 최종 상태를 출력에 더해서 반환할
    것인지 여부.
- __go_backwards__: 불리언 (기본은 False).
    True이면 입력 열을 역방향으로 처리하고
    뒤집힌 열을 반환한다.
- __stateful__: 불리언 (기본은 False). True이면
    배치에서 i 번째 표본의 마지막 상태가 각각 이어지는
    배치에서 i 번째 표본의 초기 상태로 쓰이게 된다.
- __unroll__: 불리언 (기본은 False).
    True이면 망을 전개하고
    아니면 symbolic 루프를 쓴다.
    전개를 하면 RNN의 속도를 높일 수 있지만
    메모리를 더 소모하게 된다.
    짧은 열에만 전개가 적합하다.

__참고 자료__

- [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (1997년 원저 논문)
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py#L779)</span>
### ConvLSTM2D

```python
keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
```

합성곱 LSTM.

LSTM 층과 비슷하되 입력 변형과 순환 변형이
모두 합성곱 방식이다.

__인자__

- __filters__: 정수. 출력 공간의 차원수.
    (즉 합성곱에서 출력 필터의 수)
- __kernel_size__: 정수 또는 n 개 정수로 된 튜플/리스트.
    합성곱 윈도의 차원을 지정.
- __strides__: 정수 또는 n 개 정수로 된 튜플/리스트.
    합성곱의 보폭을 지정.
    1 아닌 보폭 값을 지정하는 것과
    1 아닌 `dilation_rate` 값을 지정하는 것은
    호환되지 않음.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __data_format__: 문자열.
    "channels_last"(기본값) 또는 "channels_first".
    입력 내에서 차원들의 순서.
    `channels_last`는 `(batch, time, ..., channels)`
    형태의 입력에 해당하고
    `channels_first`는 `(batch, time, channels, ...)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
- __dilation_rate__: 정수 또는 n 개 정수로 된 튜플/리스트.
    팽창 합성곱에 사용할 팽창 비율을 지정.
    1 아닌 `dilation_rate` 값을 지정하는 것과
    1 아닌 `strides` 값을 지정하는 것은
    현재 호환되지 않음.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    아무것도 지정하지 않으면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __recurrent_activation__: 순환 단계에
    사용할 활성 함수
    ([활성](../activations.md) 참고).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __unit_forget_bias__: 불리언.
    True이면 초기화 시 망각 게이트 편향에 1을 더한다.
    `bias_initializer="zeros"`와 조합해서 사용하라.
    [Jozefowicz 외의 논문](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)에서 이를 권장한다.
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __return_sequences__: 불리언. 출력 열에 최종 출력을 반환할 것인지
    아니면 전체 열을 반환할 것인지.
- __go_backwards__: 불리언 (기본은 False).
    True이면 입력 열을 역방향으로 처리한다.
- __stateful__: 불리언 (기본은 False). True이면
    배치에서 i 번째 표본의 마지막 상태가 각각 이어지는
    배치에서 i 번째 표본의 초기 상태로 쓰이게 된다.
- __dropout__: 0과 1 사이의 float.
    입력의 선형 변환에 대해
    버릴 유닛들의 비율.
- __recurrent_dropout__: 0과 1 사이의 float.
    순환 상태의 선형 변환에 대해
    버릴 유닛들의 비율.

__입력 형태__

- data_format='channels_first'이면
    `(samples, time, channels, rows, cols)`
    형태의 5차원 텐서.
- data_format='channels_last'이면
    `(samples, time, rows, cols, channels)`
    형태의 5차원 텐서.

__출력 형태__

- `return_sequences`이면
     - data_format='channels_first'이면
        `(samples, time, filters, output_row, output_col)`
        형태의 5차원 텐서.
     - data_format='channels_last'이면
        `(samples, time, output_row, output_col, filters)`
        형태의 5차원 텐서.
- 아니면
     - data_format='channels_first'이면
        `(samples, filters, output_row, output_col)`
        형태의 4차원 텐서.
     - data_format='channels_last'이면
        `(samples, output_row, output_col, filters)`
        형태의 4차원 텐서.

    o_row과 o_col은 필터 및 패딩의 형태에 따라
    달라짐.

__예외__

- __ValueError__: 생성자 인자가 유효하지 않은 경우.

__참고 자료__

- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
현재 구현에는 셀 출력에 대한 피드백 루프가 포함돼 있지 않음

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L743)</span>
### SimpleRNNCell

```python
keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

SimpleRNN의 셀 클래스.

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 쌍곡탄젠트(`tanh`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __dropout__: 0과 1 사이의 float.
    입력의 선형 변환에 대해
    버릴 유닛들의 비율.
- __recurrent_dropout__: 0과 1 사이의 float.
    순환 상태의 선형 변환에 대해
    버릴 유닛들의 비율.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1115)</span>
### GRUCell

```python
keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=False)
```

GRU 층의 셀 클래스.

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 쌍곡탄젠트(`tanh`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __recurrent_activation__: 순환 단계에
    사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 하드 시그모이드 (`hard_sigmoid`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __dropout__: 0과 1 사이의 float.
    입력의 선형 변환에 대해
    버릴 유닛들의 비율.
- __recurrent_dropout__: 0과 1 사이의 float.
    순환 상태의 선형 변환에 대해
    버릴 유닛들의 비율.
- __implementation__: 구현 모드. 1 또는 2.
    모드 1에서는 연산들을 많은 수의
    작은 도트곱과 덧셈들로 조직하는 반면
    모드 2에서는 적은 수의 큰 연산들로 묶는다.
    두 모드는 하드웨어와 응용에 따라
    상이한 성능 특성을 보이게 된다.
- __reset_after__: GRU 동작 방식 (리셋 게이트를 행렬 곱셈 후에
    적용할지 전에 적용할지). False = "전에 적용" (기본값),
    True = "후에 적용" (CuDNN 호환).

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1689)</span>
### LSTMCell

```python
keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```

LSTM 층의 셀 클래스.

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __activation__: 사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 쌍곡탄젠트(`tanh`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __recurrent_activation__: 순환 단계에
    사용할 활성 함수
    ([활성](../activations.md) 참고).
    기본값: 하드 시그모이드 (`hard_sigmoid`).
    `None`을 주면 아무 활성도 적용하지 않음
    (즉 "선형" 활성: `a(x) = x`).
- __use_bias__: 불리언. 층에서 편향 벡터를 쓸지 여부.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __unit_forget_bias__: 불리언.
    True이면 초기화 시 망각 게이트 편향에 1을 더한다.
    참으로 설정하면 `bias_initializer="zeros"`도 써야 한다.
    [Jozefowicz 외의 논문](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)에서 이를 권장한다.
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __dropout__: 0과 1 사이의 float.
    입력의 선형 변환에 대해
    버릴 유닛들의 비율.
- __recurrent_dropout__: 0과 1 사이의 float.
    순환 상태의 선형 변환에 대해
    버릴 유닛들의 비율.
- __implementation__: 구현 모드. 1 또는 2.
    모드 1에서는 연산들을 많은 수의
    작은 도트곱과 덧셈들로 조직하는 반면
    모드 2에서는 적은 수의 큰 연산들로 묶는다.
    두 모드는 하드웨어와 응용에 따라
    상이한 성능 특성을 보이게 된다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L135)</span>
### CuDNNGRU

```python
keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

[CuDNN](https://developer.nvidia.com/cudnn) 기반의 빠른 GRU 구현.

텐서플로우 백엔드로 GPU 상에서만 돌릴 수 있다.

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __return_sequences__: 불리언. 출력 열에 최종 출력을 반환할
    것인지 아니면 전체 열을 반환할 것인지.
- __return_state__: 불리언. 최종 상태를 출력에 더해서 반환할
    것인지 여부.
- __stateful__: 불리언 (기본은 False). True이면
    배치에서 i 번째 표본의 마지막 상태가 각각 이어지는
    배치에서 i 번째 표본의 초기 상태로 쓰이게 된다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L324)</span>
### CuDNNLSTM

```python
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

[CuDNN](https://developer.nvidia.com/cudnn) 기반의 빠른 LSTM 구현.

텐서플로우 백엔드로 GPU 상에서만 돌릴 수 있다.

__인자__

- __units__: 양의 정수. 출력 공간의 차원수.
- __kernel_initializer__: 입력의 선형 변환에 쓰는
    `kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __unit_forget_bias__: 불리언.
    True이면 초기화 시 망각 게이트 편향에 1을 더한다.
    참으로 설정하면 `bias_initializer="zeros"`도 써야 한다.
    [Jozefowicz 외의 논문](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)에서 이를 권장한다.
- __recurrent_initializer__: 순환 상태의 선형 변환에 쓰는
    `recurrent_kernel` 가중치 행렬의 initializer
    ([초기화](../initializers.md) 참고).
- __bias_initializer__: 편향 벡터의 initializer
    ([초기화](../initializers.md) 참고).
- __kernel_regularizer__: `kernel` 가중치 행렬에
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __recurrent_regularizer__: `recurrent_kernel`
    가중치 행렬에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __bias_regularizer__: 편향 벡터에 적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __activity_regularizer__: 층의 출력에 ("활성"에)
    적용하는 정칙화 함수
    ([정칙화](../regularizers.md) 참고).
- __kernel_constraint__: `kernel` 가중치 행렬에
    적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __recurrent_constraint__: `recurrent_kernel`
    가중치 행렬에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __bias_constraint__: 편향 벡터에 적용하는 제약 함수
    ([제약](../constraints.md) 참고).
- __return_sequences__: 불리언. 출력 열에 최종 출력을 반환할
    것인지 아니면 전체 열을 반환할 것인지.
- __return_state__: 불리언. 최종 상태를 출력에 더해서 반환할
    것인지 여부.
- __stateful__: 불리언 (기본은 False). True이면
    배치에서 i 번째 표본의 마지막 상태가 각각 이어지는
    배치에서 i 번째 표본의 초기 상태로 쓰이게 된다.
