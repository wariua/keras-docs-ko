<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py#L253)</span>
### TimeseriesGenerator

```python
keras.preprocessing.sequence.TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)
```

시간 데이터 배치들을 생성하기 위한 유틸리티 클래스.

이 클래스는 동일 간격으로 수집한 데이터 포인트 열에 더해
보폭, 히스토리 길이 등과 같은 시계열 매개변수들을 받아서
훈련/검증을 위한 배치들을 만들어 낸다.

__인자__

- __data__: 인덱스 접근이 가능한 (리스트나 Numpy 배열 같은)
    일련의 데이터 포인트(timestep)들을 담은 제너레이터.
    데이터가 2차원이어야 하고 0번 축이
    시간 차원이라고 기대한다.
- __targets__: `data`의 timestep들에 대응하는 목표들.
    `data`와 같은 길이여야 한다.
- __length__: 출력 열들의 길이 (timestep 수).
- __sampling_rate__: 열 내에서 연속된 개별 timestep들
    사이의 주기. 그 간격이 `r`이면 timestep
    `data[i]`, `data[i-r]`, ..., `data[i - length]`를
    써서 표본 열을 만든다.
- __stride__: 연속된 출력 열들 사이의 주기.
    보폭이 `s`이면 연속하는 출력 표본들이
    `data[i]`, `data[i+s]`, `data[i+2*s]` 등을
    중심으로 하게 된다.
- __start_index__: `start_index` 전의 데이터 포인트들은
    출력 열에 쓰이지 않게 된다. 데이터 일부를 테스트나
    검증용으로 떼어놓는 데 유용하다.
- __end_index__: `end_index` 후의 데이터 포인터들은
    출력 열에 쓰이지 않게 된다. 데이터 일부를 테스트나
    검증용으로 떼어놓는 데 유용하다.
- __shuffle__: 출력 표본들을 뒤섞을지 여부.
    아니면 연대순으로 뽑아냄.
- __reverse__: 불리언. `true`이면 각 출력 표본 내의 timestep들이
    연대역순이 됨.
- __batch_size__: 각 배치(마지막은 제외) 안의 시계열 표본 수.

__반환__

[Sequence](/utils/#sequence) 인스턴스.

__예시__


```python
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=2,
                               batch_size=2)
assert len(data_gen) == 20

batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                      np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                      np.array([[10], [11]]))
```

----

### pad_sequences


```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
```


패딩으로 열들을 같은 길이로 만들기.

이 함수는 `num_samples`짜리 열(장수들의 리스트)들의
리스트를 `(num_samples, num_timesteps)` 형태의
2차원 Numpy 배열로 변환한다.
`num_timesteps`는 `maxlen` 인자를 주면 그 값이고
아니면 가장 긴 열의 길이이다.

열이 `num_timesteps`보다 짧으면
끝에 `value`로 패딩을 붙인다.

열이 `num_timesteps`보다 길면 끝을 잘라서
원하는 길이에 맞도록 한다.
패딩이나 자르기가 어느 쪽에서 이뤄지는지는
각각 `padding`과 `truncating` 인자에 의해 정해진다.

앞쪽 패딩이 기본이다.

__인자__

- __sequences__: 리스트들의 리스트. 각 항목이 열이다.
- __maxlen__: 정수. 모든 열들의 최대 길이.
- __dtype__: 출력 열들의 타입.
- __padding__: 문자열. 'pre' 또는 'post'.
    각 열의 앞과 뒤 어느 쪽에 패딩을 추가할 것인가.
- __truncating__: 문자열. 'pre' 또는 'post'.
    열이 `maxlen`보다 길면 열의 시작 또는 끝에서
    값들을 제거한다.
- __value__: 실수. 패딩 값.

__반환__

- __x__: `(len(sequences), maxlen)` 형태의 Numpy 배열

__Raises__

- __ValueError__: `truncating`이나 `padding`의 값이 유효하지 않은 경우,
    또는 `sequences` 항목의 형태가 유효하지 않은 경우.

----

### skipgrams


```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size, window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
```


Skipgram 단어 쌍들을 생성.

이 함수는 단어 인덱스들의 열(정수들의 리스트)을
다음 형태의 단어 튜플들로 변환한다.

- (단어, 동일 윈도 내의 단어), 레이블 1 (양성 표본)
- (단어, 어휘에서 뽑은 임의 단어), 레이블 0 (음성 표본)

Skipgram에 대해선 Mikolov 외의 이 금언적인 논문을 읽어 보라:
[Efficient Estimation of Word Representations in
Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

__인자__

- __sequence__: 단어 인덱스(정수)들의 리스트로 표현된
    단어 열(문장). `sampling_table`을 쓰는 경우
    단어 인덱스들이 참조 데이터셋 내에서 단어들의
    순위와 일치한다고 기대한다. (가령 인덱스 10이
    10번째로 자주 등장하는 토큰을 표현.)
    참고로 인덱스 0은 단어가 아니라고 기대하며 건너뛴다.
- __vocabulary_size__: int. 가능한 최대 단어 인덱스 + 1
- __window_size__: int. 샘플링 윈도(정확히는 원도 절반)의 크기.
    단어 `w_i`의 윈도가
    `[i - window_size, i + window_size+1]`이 된다.
- __negative_samples__: float >= 0. 0이면 음성 (임의) 표본 없음.
    1이면 양성 표본과 같은 수만큼.
- __shuffle__: 단어 짝들을 반환 전에 뒤섞을지 여부.
- __categorical__: bool. `False`이면 레이블이 정수가 된다.
    가령 `[0, 1, 1 .. ]`.
    `True`이면 레이블이 범주 방식이 된다.
    가령 `[[1,0],[0,1],[0,1] .. ]`.
- __sampling_table__: `vocabulary_size` 크기의 1차원 배열.
    항목 i는 순위 i인 단어를 표본화할 확률을 나타냄.
- __seed__: 난수 시드.

__반환__

couples, labels: `couples`는 int 짝들이고
    `labels`는 0 또는 1.

__주의__

관행상 어휘에서 인덱스 0은 단어가 아니며
건너뛰게 된다.

----

### make_sampling_table


```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-05)
```


순위 기반의 표본화 확률 테이블을 생성.

`skipgrams`의 `sampling_table` 인자 생성에 사용한다.
`sampling_table[i]`는 데이터셋에서 i 번째
흔한 단어를 표본으로 뽑을 확률이다.
(균형을 위해선 더 흔한 단어를 더 드물게 표본으로 뽑아야 한다.)

word2vec에서 쓰는 표본 분포에 따라
표본화 확률을 만들어 낸다.

```
p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
    (word_frequency / sampling_factor)))
```

단어 빈도가 지프의 법칙(s=1)을 따른다고 가정하고
빈도(순위)의 근사 수치를 얻는다.

`frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
(`gamma`는 오일러-마스케로니 상수)

__인자__

- __size__: int. 표본화 가능한 단어들의 수.
- __sampling_factor__: word2vec 식의 표본화 계수.

__반환__

길이가 `size`인 1차원 Numpy 배열.
i 번째 항목은 순위 i인 단어를 표본화할 확률임.
