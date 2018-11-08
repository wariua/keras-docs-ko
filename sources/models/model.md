# Model 클래스 API

함수형 API 방식에서는 어떤 입력 텐서와 출력 텐서가 주어졌을 때 다음과 같이 `Model` 인스턴스를 만들 수 있다.

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

`a`가 주어질 때 `b`를 계산하는 데 필요한 층들이 모두 이 모델에 포함된다.

다입력 내지 다출력 모델인 경우에는 리스트를 쓸 수도 있다.

```python
model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
```

`Model`로 할 수 있는 것들에 대한 자세한 소개는 [케라스 함수형 API 소개](/getting-started/functional-api-guide)를 읽어 보라.


## 메소드

### compile


```python
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```


모델 훈련 방식을 설정한다.

__인자__

- __optimizer__: 문자열(옵티마이저 이름) 또는 옵티마이저 인스턴스.
    [옵티마이저](/optimizers) 참고.
- __loss__: 문자열(목표 함수 이름) 또는 목표 함수.
    [손실](/losses) 참고.
    모델 출력이 여러 개이면 손실들의 딕셔너리 내지 리스트를 줘서
    각 출력마다 다른 손실을 쓸 수 있다.
    그때 모델에서 최소화 하게 되는 손실 값은
    개별 손실 모두의 합이다.
- __metrics__: 훈련 및 테스트 중에 모델에서 평가하는
    지표들의 리스트.
    보통은 `metrics=['accuracy']`를 쓰게 된다.
    다출력 모델에서 출력별로 다른 지표를 지정하려면
    `metrics={'output_a': 'accuracy'}`처럼
    딕셔너리를 줄 수 있다.
- __loss_weights__: 선택적. 여러 모델 출력들의 손실 기여에
    가중치를 주기 위한 스칼라 계수(파이썬 float)들을 나타내는
    리스트 내지 딕셔너리.
    그러면 모델에서 최소화 하게 되는 손실 값은
    `loss_weights` 계수들로 가중치를 준
    개별 손실 모두의 *가중치 합*이 된다.
    리스트이면 모델의 출력들과 1:1 대응되기를 기대한다.
    딕셔너리이면 출력 이름(문자열)을 스칼라 계수로
    매핑 하기를 기대한다.
- __sample_weight_mode__: timestep별로 표본에 가중치를
    줄 필요가 있다면 (2D 가중치) `"temporal"`로 설정하라.
    `None`으로 하면 표본별 가중치(1D)이다.
    모델에 입력이 여러 개이면 모드들의
    딕셔너리 내지 리스트를 줘서 각 입력마다 다른
    `sample_weight_mode`를 쓸 수 있다.
- __weighted_metrics__: 훈련 및 테스트 중에 sample_weight이나
    class_weight으로 평가해서 가중치를 주는 지표들의 리스트.
- __target_tensors__: 기본적으로 케라스에서는 모델 목표를 위한
    플레이스홀더를 만들고 훈련 동안 목표 데이터를 집어넣게
    된다. 그렇게 하는 대신 자기만의 목표 텐서를 쓰고 싶다면
    (그러면 케라스에서는 훈련 시점에 그 목표들에 대한
    외부 Numpy 데이터를 기대하지 않게 된다.) `target_tensors`
    인자를 통해 지정할 수 있다. 단일 텐서일 수도 있고
    (단일 출력 모델), 텐서들의 리스트거나 출력 이름에서
    목표 텐서로 가는 dict 매핑일 수도 있다.
- __**kwargs__: 테아노/CNTK 백엔드 사용 시
    이 인자들이 `K.function`으로 전달된다.
    텐서플로우 백엔드 사용 시
    이 인자들이 `tf.Session.run`으로 전달된다.

__예외__

- __ValueError__: `optimizer`, `loss`, `metrics`, `sample_weight_mode`
    인자가 유효하지 않은 경우.

----

### fit


```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```


지정한 에포크 수(데이터셋 반복 횟수)만큼 모델을 훈련시킨다.

__인자__

- __x__: 훈련 데이터의 Numpy 배열이거나 (모델 입력이 하나인 경우),
    Numpy 배열들의 리스트 (모델 입력이 여럿인 경우).
    모델의 입력 층들에 이름이 있으면 입력 이름을 Numpy 배열로
    매핑 하는 딕셔너리를 줄 수도 있다.
    프레임워크 자체 텐서(가령 텐서플로우 데이터 텐서)를 가지고
    넣어 주는 경우 `x`가 `None`(기본값)일 수 있다.
- __y__: 목표(레이블) 데이터의 Numpy 배열이거나
    (모델 출력이 하나인 경우),
    Numpy 배열들의 리스트 (모델 출력이 여럿인 경우).
    모델 출력 층들에 이름이 있으면 출력 이름을 Numpy 배열로
    매핑 하는 딕셔너리를 줄 수도 있다.
    프레임워크 자체 텐서(가령 텐서플로우 데이터 텐서)를 가지고
    넣어 주는 경우 `y`가 `None`(기본값)일 수 있다.
- __batch_size__: 정수 또는 `None`.
    경사 갱신 한 번당 표본 수.
    지정하지 않으면 `batch_size`로 32를 쓴다.
- __epochs__: 정수. 모델을 훈련시킬 에포크 수.
    에포크는 주어진 데이터 `x` 및 `y` 전체에 대한
    반복 한 세트이다.
    참고로 `initial_epoch`를 같이 생각하면
    `epochs`를 "마지막 에포크"로 이해해야 한다.
    즉 `epochs`로 준 반복 횟수만큼 모델을 훈련시키는
    게 아니라 에포크 인덱스가 `epochs`가 될 때까지
    훈련시키는 것이다.
- __verbose__: 정수. 0, 1, 2. 출력 상세 정도.
    0 = 조용하게, 1 = 진행 막대, 2 = 에포크마다 한 줄씩.
- __callbacks__: `keras.callbacks.Callback` 인스턴스들의 리스트.
    훈련 동안 적용할 콜백들의 목록.
    [콜백](/callbacks) 참고.
- __validation_split__: 0과 1 사이 float.
    데이터 검증에 사용할 훈련 데이터의 비율.
    모델에서 훈련 데이터의 그 부분을 떼어놓고서
    그걸로는 훈련을 하지 않으며,
    각 에포크 끝에서 그 데이터를 가지고
    손실과 모델 지표들을 평가하게 된다.
    주어진 `x` 및 `y` 데이터의 뒤섞기 전의
    뒤쪽 표본들을 검증 데이터로 선택한다.
- __validation_data__: 튜플 `(x_val, y_val)` 또는
    튜플 `(x_val, y_val, val_sample_weights)`. 각 에포크 끝에서
    이 데이터에 대해 손실과 모델 지표들을 평가한다.
    이 데이터에 대해선 모델을 훈련시키지 않는다.
    `validation_data`를 주면 `validation_split`은 무시한다.
- __shuffle__: bool (각 에포크 전에 훈련 데이터를
    뒤섞을지 여부) 또는 str ('batch').
    'batch'는 HDF5 데이터의 제약에 대처하기 위한
    특별한 옵션인데, 배치 크기 덩어리에서 뒤섞기를 한다.
    `steps_per_epoch`가 `None`이 아닐 때는 아무 효력이 없다.
- __class_weight__: 선택적. 클래스 인덱스(정수)에서
    가중치(실수) 값으로의 딕셔너리 매핑이며,
    (훈련 중에 한해서) 손실 함수에 가중치를 주는 데
    쓰인다. 대표가 모자란 클래스의 표본들에 대해 모델에서
    "더 신경을 쓰게" 만드는 데 유용할 수 있다.
- __sample_weight__: 선택적. 훈련 표본들에 대한 가중치들의
    Numpy 배열이며, (훈련 중에 한해서) 손실 함수에
    가중치를 주는 데 쓰인다. 입력 표본과 길이가 같은
    길다란 (1차원) Numpy 배열을 줄 수도 있고
    (가중치와 표본이 1:1 대응),
    temporal 데이터인 경우에는
    `(samples, sequence_length)`
    형태의 2D 배열을 줘서 각 표본의 timestep마다
    다른 가중치를 적용할 수 있다.
    그 경우 `compile()`에서
    `sample_weight_mode="temporal"`을
    꼭 지정해 줘야 할 것이다.
- __initial_epoch__: 정수.
    훈련을 시작할 에포크.
    (이전 훈련 돌리던 걸 재개하는 데 유용하다.)
- __steps_per_epoch__: 정수 또는 `None`.
    한 에포크당 단계(표본 배치)들의 수.
    텐서플로우 데이터 텐서 같은 입력 텐서로
    훈련을 시킬 때 기본값 `None`은
    데이터셋의 샘플 수를 배치 크기로 나눈 것이다.
    그 값을 알아낼 수 없으면 1이다.
- __validation_steps__: `steps_per_epoch`를 지정한 경우에만
    의미가 있다. 단계(표본 배치)들을 이 수만큼
    검증한 다음 멈춘다.

__반환__

`History` 객체. `History.history` 속성은
연속된 에포크에서의 훈련 손실 값과 지표 값,
그리고 (적용 가능한 경우) 검증 손실 값과
검증 지표 값을 기록한 것이다.

__예외__

- __RuntimeError__: 모델을 한번도 컴파일 하지 않은 경우.
- __ValueError__: 제공한 입력 데이터와 모델의 기대 사이에
    불일치가 있는 경우.

----

### evaluate


```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
```


테스트 모드인 모델에 대해서 손실 값과 지표 값을 반환한다.

배치 단위로 계산이 이뤄진다.

__인자__

- __x__: 테스트 데이터의 Numpy 배열이거나 (모델 입력이 하나인 경우),
    Numpy 배열들의 리스트 (모델 입력이 여럿인 경우).
    모델의 입력 층들에 이름이 있으면 입력 이름을 Numpy 배열로
    매핑 하는 딕셔너리를 줄 수도 있다.
    프레임워크 자체 텐서(가령 텐서플로우 데이터 텐서)를 가지고
    넣어 주는 경우 `x`가 `None`(기본값)일 수 있다.
- __y__: 목표(레이블) 데이터의 Numpy 배열이거나
    (모델 출력이 하나인 경우),
    Numpy 배열들의 리스트 (모델 출력이 여럿인 경우).
    모델 출력 층들에 이름이 있으면 출력 이름을 Numpy 배열로
    매핑 하는 딕셔너리를 줄 수도 있다.
    프레임워크 자체 텐서(가령 텐서플로우 데이터 텐서)를 가지고
    넣어 주는 경우 `y`가 `None`(기본값)일 수 있다.
- __batch_size__: 정수 또는 `None`.
    평가 단계 한 번당 표본 수.
    지정하지 않으면 `batch_size`로 32를 쓴다.
- __verbose__: 0 또는 1. 출력 상세 정도.
    0 = 조용하게, 1 = 진행 막대.
- __sample_weight__: 선택적. 테스트 표본들에 대한 가중치들의
    Numpy 배열이며, 손실 함수에 가중치를 주는 데 쓰인다.
    입력 표본과 길이가 같은
    길다란 (1차원) Numpy 배열을 줄 수도 있고
    (가중치와 표본이 1:1 대응),
    temporal 데이터인 경우에는
    `(samples, sequence_length)`
    형태의 2D 배열을 줘서 각 표본의 timestep마다
    다른 가중치를 적용할 수 있다.
    그 경우 `compile()`에서
    `sample_weight_mode="temporal"`을
    꼭 지정해 줘야 할 것이다.
- __steps__: 정수 또는 `None`.
    단계(표본 배치)들을 이 수만큼
    거친 다음 평가 과정이 끝난다.
    기본값 `None`이면 무시.

__반환__

스칼라 테스트 손실 (모델에 출력이 하나이고 지표가 없는 경우),
또는 스칼라들의 리스트 (모델에 출력 및/또는 지표가 여럿인
경우). 그 스칼라 출력에 대한 표시용 레이블을
`model.metrics_names` 속성에서 얻을 수 있다.

----

### predict


```python
predict(x, batch_size=None, verbose=0, steps=None)
```


입력 표본들에 대한 예측 출력을 만들어 낸다.

배치 단위로 계산이 이뤄진다.

__인자__

- __x__: 입력 데이터. Numpy 배열
    (또는 모델에 입력이 여러 개이면 Numpy 배열들의 리스트).
- __batch_size__: 정수. 지정하지 않으면 32를 쓴다.
- __verbose__: 출력 상세 정도, 0 또는 1.
- __steps__: 단계(표본 배치)들을 이 수만큼
    거친 다음 평가 과정이 끝난다.
    기본값 `None`이면 무시.

__반환__

예측들의 Numpy 배열(들).

__예외__

- __ValueError__: 제공한 입력 데이터와 모델의 기대 사이에
    불일치가 있는 경우,
    또는 상태 유지형 모델에서 받은 표본 개수가
    배치 크기의 배수가 아닌 경우.

----

### train_on_batch


```python
train_on_batch(x, y, sample_weight=None, class_weight=None)
```


데이터 배치 하나에 경사 갱신을 한 번 실행한다.

__인자__

- __x__: 훈련 데이터의 Numpy 배열이거나,
    모델에 입력이 여럿인 경우 Numpy 배열들의 리스트.
    모델 입력 모두에 이름이 있으면
    입력 이름을 Numpy 배열로 매핑 하는
    딕셔너리를 줄 수도 있다.
- __y__: 목표 데이터의 Numpy 배열이거나,
    모델에 출력이 여럿인 경우 Numpy 배열들의 리스트.
    모델 출력 모두에 이름이 있으면
    출력 이름을 Numpy 배열로 매핑 하는
    딕셔너리를 줄 수도 있다.
- __sample_weight__: 선택적. x와 같은 길이의 배열이며,
    각 표본에 대한 모델의 손실에 적용할 가중치를 담는다.
    temporal 데이터인 경우에는 `(samples, sequence_length)`
    형태의 2D 배열을 줘서 각 표본의 timestep마다
    다른 가중치를 적용할 수 있다.
    그 경우 `compile()`에서 `sample_weight_mode="temporal"`을
    꼭 지정해 줘야 할 것이다.
- __class_weight__: 선택적. 클래스 인덱스(정수)에서
    가중치(실수)로 가는 딕셔너리 매핑이며, 훈련 동안에
    그 클래스의 표본들에 대한 모델의 손실에 적용된다.
    대표가 모자란 클래스의 표본들에 대해 모델에서
    "더 신경을 쓰게" 만드는 데 유용할 수 있다.

__반환__

스칼라 훈련 손실
(모델에 출력이 하나이고 지표가 없는 경우),
또는 스칼라들의 리스트 (모델에 출력 및/또는 지표가
여럿인 경우). 그 스칼라 출력에 대한 표시용 레이블을
`model.metrics_names` 속성에서 얻을 수 있다.

----

### test_on_batch


```python
test_on_batch(x, y, sample_weight=None)
```


표본 배치 하나에 대해 모델을 테스트 한다.

__인자__

- __x__: 테스트 데이터의 Numpy 배열이거나,
    모델에 입력이 여럿인 경우 Numpy 배열들의 리스트.
    모델 입력 모두에 이름이 있으면
    입력 이름을 Numpy 배열로 매핑 하는
    딕셔너리를 줄 수도 있다.
- __y__: 목표 데이터의 Numpy 배열이거나,
    모델에 출력이 여럿인 경우 Numpy 배열들의 리스트.
    모델 출력 모두에 이름이 있으면
    출력 이름을 Numpy 배열로 매핑 하는
    딕셔너리를 줄 수도 있다.
- __sample_weight__: 선택적. x와 같은 길이의 배열이며,
    각 표본에 대한 모델의 손실에 적용할 가중치를 담는다.
    temporal 데이터인 경우에는 `(samples, sequence_length)`
    형태의 2D 배열을 줘서 각 표본의 timestep마다
    다른 가중치를 적용할 수 있다.
    그 경우 `compile()`에서 `sample_weight_mode="temporal"`을
    꼭 지정해 줘야 할 것이다.

__반환__

스칼라 테스트 손실 (모델에 출력이 하나이고 지표가 없는 경우),
또는 스칼라들의 리스트 (모델에 출력 및/또는 지표가 여럿인
경우). 그 스칼라 출력에 대한 표시용 레이블을
`model.metrics_names` 속성에서 얻을 수 있다.

----

### predict_on_batch


```python
predict_on_batch(x)
```


표본 배치 하나에 대한 예측을 반환한다.

__인자__

- __x__: 입력 표본들. Numpy 배열.

__반환__

예측들의 Numpy 배열(들).

----

### fit_generator


```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```


파이썬 제너레이터가 (또는 `Sequence` 인스턴스가) 배치 단위로 생성한 데이터에 대해 모델을 훈련시킨다.

효율성을 위해 모델과 병렬로 제너레이터를 돌린다.
그래서 예를 들어 GPU에서 모델을 훈련시키는 것과 병렬로
CPU에서 이미지에 대한 실시간 데이터 증대를 할 수 있다.

`keras.utils.Sequence`를 사용하면 순서가 보장되며
`use_multiprocessing=True`를 쓸 때 에포크당 각 입력이
한 번씩만 쓰인다고 보장된다.

__인자__

- __generator__: 제너레이터, 또는 멀티프로세싱
    사용 시 데이터 중복을 피하려면
    `Sequence`(`keras.utils.Sequence`) 객체 인스턴스.
    제너레이터의 출력은 다음 중 하나여야 한다.
    - 튜플 `(inputs, targets)`
    - 튜플 `(inputs, targets, sample_weights)`

    이 튜플(제너레이터의 입력 한 개)이 배치 한 개를 이룬다.
    따라서 튜플 내의 모든 배열들의 길이가 같아야 한다.
    (배치 크기와 같아야 한다.) 서로 다른 배치들은 크기가
    다를 수 있다. 예를 들어 데이터셋 크기가 배치 크기로
    나눠 떨어지지 않으면 에포크 마지막 배치가
    다른 배치보다 작다.
    제너레이터는 자기 데이터 상에서 무한히 맴돌면 된다.
    모델에서 배치를 `steps_per_epoch` 개 보고 나면
    에포크가 끝난다.

- __steps_per_epoch__: 정수.
    한 에포크당 `generator`에서 yield 하는
    단계(표본 배치)들의 수. 보통은 데이터셋의 표본 수를
    배치 크기로 나눈 것과 같을 것이다.
    `Sequence`에서: 지정하지 않으면
    `len(generator)`를 단계 수로 쓴다.
- __epochs__: 정수. 모델을 훈련시킬 에포크 수.
    에포크는 주어진 데이터 전체에 대한
    `steps_per_epoch`로 정의된 반복 한 세트이다.
    참고로 `initial_epoch`를 같이 생각하면
    `epochs`를 "마지막 에포크"로 이해해야 한다.
    즉 `epochs`로 준 반복 횟수만큼 모델을 훈련시키는
    게 아니라 에포크 인덱스가 `epochs`가 될 때까지
    훈련시키는 것이다.
- __verbose__: 정수. 0, 1, 2. 출력 상세 정도.
    0 = 조용하게, 1 = 진행 막대, 2 = 에포크마다 한 줄씩.
- __callbacks__: `keras.callbacks.Callback` 인스턴스들의 리스트.
    훈련 동안 적용할 콜백들의 목록.
    [콜백](/callbacks) 참고.
- __validation_data__: 다음 중 하나일 수 있다.
    - 검증 데이터의 제너레이터 내지 `Sequence` 객체
    - 튜플 `(x_val, y_val)`
    - 튜플 `(x_val, y_val, val_sample_weights)`

    각 에포크 끝에서
    이 데이터에 대해 손실과 모델 지표들을 평가한다.
    이 데이터에 대해선 모델을 훈련시키지 않는다.

- __validation_steps__: `validation_data`가 제너레이터인
    경우에만 의미가 있다. `validation_data` 제너레이터에서
    단계(표본 배치)들을 이 수만큼 yield 한 다음
    에포크 끝에서 멈춘다. 보통은 검증 데이터셋의
    표본 수를 배치 크기로 나눈 것과 같을 것이다.
    `Sequence`에서: 지정하지 않으면
    `len(validation_data)`를 단계 수로 쓴다.
- __class_weight__: 선택적. 클래스 인덱스(정수)에서
    가중치(실수) 값으로의 딕셔너리 매핑이며,
    (훈련 중에 한해서) 손실 함수에 가중치를 주는 데
    쓰인다. 대표가 모자란 클래스의 표본들에 대해 모델에서
    "더 신경을 쓰게" 만드는 데 유용할 수 있다.
- __max_queue_size__: 정수. 제너레이터 큐 최대 크기.
    지정하지 않으면 `max_queue_size`로 10을 쓴다.
- __workers__: 정수. 프로세스 기반 스레딩 사용 시
    돌릴 프로세스 최대 개수.
    지정하지 않으면 `workers` 기본값은 1이다.
    0이면 메인 스레드에서 제너레이터를 실행한다.
- __use_multiprocessing__: bool.
    `True`이면 프로세스 기반 스레딩을 쓴다.
    지정하지 않으면 `use_multiprocessing` 기본값은 `False`이다.
    참고로 그 구현이 다중 프로세스 방식이기 때문에
    pickle 불가능한 인자를 제너레이터에
    주지 않는 게 좋다. 자식 프로세스들에게
    쉽게 전달할 수 없기 때문이다.
- __shuffle__: bool. 각 에포크 시작 때 배치들의 순서를
    뒤섞을지 여부. `Sequence`(`keras.utils.Sequence`)의
    인스턴스에서만 쓴다.
    `steps_per_epoch`가 `None`이 아닐 때는 아무 효력이 없다.
- __initial_epoch__: 정수.
    훈련을 시작할 에포크.
    (이전 훈련 돌리던 걸 재개하는 데 유용하다.)

__반환__

`History` 객체. `History.history` 속성은
연속된 에포크에서의 훈련 손실 값과 지표 값,
그리고 (적용 가능한 경우) 검증 손실 값과
검증 지표 값을 기록한 것이다.

__예외__

- __ValueError__: 제너레이터가 유효하지 않은 형식의 데이터를 내놓은 경우.

__예시__


```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # 파일의 각 행을 읽어서 입력 데이터와
                # 레이블의 numpy 배열들 생성
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)
```

----

### evaluate_generator


```python
evaluate_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```


데이터 제너레이터에서 모델을 평가한다.

제너레이터는 `test_on_batch`에서 받는 것과 같은
종류의 데이터를 반환해야 한다.

__인자__

- __generator__: 튜플 `(inputs, targets)`나
    `(inputs, targets, sample_weights)`를 내놓는 제너레이터,
    또는 멀티프로세싱 사용 시 데이터 중복을 피하려면
    `Sequence`(`keras.utils.Sequence`) 객체 인스턴스.
- __steps__: `generator`에서 단계(표본 배치)들을
    이 수만큼 yield 한 다음 멈춘다.
    `Sequence`에서: 지정하지 않으면
    `len(generator)`를 단계 수로 쓴다.
- __max_queue_size__: 제너레이터 큐 최대 크기.
- __workers__: 정수. 프로세스 기반 스레딩 사용 시
    돌릴 프로세스 최대 개수.
    지정하지 않으면 `workers` 기본값은 1이다.
    0이면 메인 스레드에서 제너레이터를 실행한다.
- __use_multiprocessing__: `True`이면 프로세스 기반 스레딩을 쓴다.
    참고로 그 구현이 다중 프로세스 방식이기 때문에
    pickle 불가능한 인자를 제너레이터에
    주지 않는 게 좋다. 자식 프로세스들에게
    쉽게 전달할 수 없기 때문이다.
- __verbose__: 출력 상세 정도, 0 또는 1.

__반환__

스칼라 테스트 손실 (모델에 출력이 하나이고 지표가 없는 경우),
또는 스칼라들의 리스트 (모델에 출력 및/또는 지표가 여럿인
경우). 그 스칼라 출력에 대한 표시용 레이블을
`model.metrics_names` 속성에서 얻을 수 있다.

__예외__

- __ValueError__: 제너레이터가 유효하지 않은 형식의 데이터를 내놓은 경우.

----

### predict_generator


```python
predict_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```


데이터 제너레이터에서 온 입력 표본에 대해 예측을 만들어 낸다.

제너레이터는 `predict_on_batch`에서 받는 것과 같은
종류의 데이터를 반환해야 한다.

__인자__

- __generator__: 입력 표본들의 배치를 내놓는 제너레이터,
    또는 멀티프로세싱 사용 시 데이터 중복을 피하려면
    `Sequence`(`keras.utils.Sequence`) 객체 인스턴스.
- __steps__: `generator`에서 단계(표본 배치)들을
    이 수만큼 yield 한 다음 멈춘다.
    `Sequence`에서: 지정하지 않으면
    `len(generator)`를 단계 수로 쓴다.
- __max_queue_size__: 제너레이터 큐 최대 크기.
- __workers__: 정수. 프로세스 기반 스레딩 사용 시
    돌릴 프로세스 최대 개수.
    지정하지 않으면 `workers` 기본값은 1이다.
    0이면 메인 스레드에서 제너레이터를 실행한다.
- __use_multiprocessing__: `True`이면 프로세스 기반 스레딩을 쓴다.
    참고로 그 구현이 다중 프로세스 방식이기 때문에
    pickle 불가능한 인자를 제너레이터에
    주지 않는 게 좋다. 자식 프로세스들에게
    쉽게 전달할 수 없기 때문이다.
- __verbose__: 출력 상세 정도, 0 또는 1.

__반환__

예측들의 Numpy 배열(들).

__예외__

- __ValueError__: 제너레이터가 유효하지 않은 형식의 데이터를 내놓은 경우.

----

### get_layer


```python
get_layer(name=None, index=None)
```


(유일한) 이름이나 인덱스를 가지고 층을 얻어 온다.

`name`과 `index` 둘 다 주면 `index`를 우선하게 된다.

인덱스는 너비 우선 그래프 순회 (상향) 순서에 따른다.

__인자__

- __name__: 문자열, 층 이름.
- __index__: 정수, 층 인덱스.

__반환__

층 인스턴스.

__예외__

- __ValueError__: 층 이름이나 인덱스가 유효하지 않은 경우.

