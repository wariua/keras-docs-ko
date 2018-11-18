<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L193)</span>
### Add

```python
keras.layers.Add()
```

입력들의 목록으로 덧셈을 하는 층.

모두 같은 형태인 텐서들의 리스트를
입력으로 받아서 (역시 같은 형태인)
텐서 하나를 반환한다.

__예시__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.Add()([x1, x2])  # added = keras.layers.add([x1, x2])와 동등

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L223)</span>
### Subtract

```python
keras.layers.Subtract()
```

두 입력으로 빼기를 하는 층.

형태가 같은 텐서들의 크기 2인 리스트를
입력으로 받아서 역시 같은 형태인
텐서 (inputs[0] - inputs[1])를 반환한다.

__예시__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# subtracted = keras.layers.subtract([x1, x2])와 동등
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L260)</span>
### Multiply

```python
keras.layers.Multiply()
```

입력들의 목록으로 (항목별) 곱셈을 하는 층.

모두 같은 형태인 텐서들의 리스트를
입력으로 받아서 (역시 같은 형태인)
텐서 하나를 반환한다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L275)</span>
### Average

```python
keras.layers.Average()
```

입력들의 목록으로 평균을 내는 층.

모두 같은 형태인 텐서들의 리스트를
입력으로 받아서 (역시 같은 형태인)
텐서 하나를 반환한다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L290)</span>
### Maximum

```python
keras.layers.Maximum()
```

입력들의 목록으로 (항목별) 최댓값을 계산하는 층.

모두 같은 형태인 텐서들의 리스트를
입력으로 받아서 (역시 같은 형태인)
텐서 하나를 반환한다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L320)</span>
### Concatenate

```python
keras.layers.Concatenate(axis=-1)
```

입력들의 목록을 이어 붙이는 층.

이어 붙이기 축을 제외하고 모두 같은 형태인
텐서들의 리스트를 입력으로 받아서
모든 입력을 접합한 텐서 하나를 반환한다.

__인자__

- __axis__: 이어 붙여 나가는 축.
- __**kwargs__: 층 표준 키워드 인자들.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L408)</span>
### Dot

```python
keras.layers.Dot(axes, normalize=False)
```

두 텐서 표본들 간의 도트곱을 계산하는 층.

가령 형태가 `(batch_size, n)`인 두 텐서 `a`와 `b`의 리스트에 적용하면
출력은 형태가 `(batch_size, 1)`인 텐서 하나가 되고
각 항목 `i`는 `a[i]`와 `b[i]`의 도트곱이 된다.

__인자__

- __axes__: 정수 또는 정수들의 튜플.
    이 축(들)을 따라 도트곱을 취함.
- __normalize__: 도트곱을 취하기 전에 도트곱
    축을 따라 표본들을 L2 정규화 할지 여부.
    True로 설정 시 도트곱의 출력은
    두 표본 간의 코사인 근접도이다.
- __**kwargs__: 층 표준 키워드 인자들.

----

### add


```python
keras.layers.add(inputs)
```


`Add` 층의 함수 인터페이스.

__인자__

- __inputs__: (최소 2개인) 입력 텐서들의 리스트.
- __**kwargs__: 층 표준 키워드 인자들.

__반환__

입력들의 합인 텐서.

__예시__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

### subtract


```python
keras.layers.subtract(inputs)
```


`Subtract` 층의 함수 인터페이스.

__인자__

- __inputs__: (정확히 2개인) 입력 텐서들의 리스트.
- __**kwargs__: 층 표준 키워드 인자들.

__반환__

입력들의 차인 텐서.

__예시__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

### multiply


```python
keras.layers.multiply(inputs)
```


`Multiply` 층의 함수 인터페이스.

__인자__

- __inputs__: (최소 2개인) 입력 텐서들의 리스트.
- __**kwargs__: 층 표준 키워드 인자들.

__반환__

입력들의 항목별 곱인 텐서.

----

### average


```python
keras.layers.average(inputs)
```


`Average` 층의 함수 인터페이스.

__인자__

- __inputs__: (최소 2개인) 입력 텐서들의 리스트.
- __**kwargs__: 층 표준 키워드 인자들.

__반환__

입력들의 평균인 텐서.

----

### maximum


```python
keras.layers.maximum(inputs)
```


`Maximum` 층의 함수 인터페이스.

__인자__

- __inputs__: (최소 2개인) 입력 텐서들의 리스트.
- __**kwargs__: 층 표준 키워드 인자들.

__반환__

입력들의 항목별 최댓값인 텐서.

----

### concatenate


```python
keras.layers.concatenate(inputs, axis=-1)
```


`Concatenate` 층의 함수 인터페이스.

__인자__

- __inputs__: (최소 2개인) 입력 텐서들의 리스트.
- __axis__: 접합 축.
- __**kwargs__: 층 표준 키워드 인자들.

__반환__

축 `axis`를 따라 입력들을 접합한 텐서.

----

### dot


```python
keras.layers.dot(inputs, axes, normalize=False)
```


`Dot` 층의 함수 인터페이스.

__인자__

- __inputs__: (최소 2개인) 입력 텐서들의 리스트.
- __axes__: 정수 또는 정수들의 튜플.
    이 축(들)을 따라 도트곱을 취함.
- __normalize__: 도트곱을 취하기 전에 도트곱
    축을 따라 표본들을 L2 정규화 할지 여부.
    True로 설정 시 도트곱의 출력은
    두 표본 간의 코사인 근접도이다.
- __**kwargs__: 층 표준 키워드 인자들.

__반환__

입력의 표본들의 도트곱인 텐서.
