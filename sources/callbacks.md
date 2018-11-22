## 콜백 사용

콜백이란 훈련 절차의 특정 단계들에서 적용할 함수들이다. 콜백을 이용하면 훈련 중에 모델의 내부 상태와 통계를 조망할 수 있다. `Sequential` 내지 `Model` 클래스의 `.fit()` 메소드로 콜백 목록을 (키워드 인자 `callbacks`로) 주면 된다. 그러면 훈련의 각 단계에서 콜백의 해당 메소드가 호출된다.

---

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L146)</span>
### Callback

```python
keras.callbacks.Callback()
```

새로운 콜백을 만드는 데 쓰는 추상적 기반 클래스.

__속성__

- __params__: dict. 훈련 매개변수들.
    (예: 출력 수준, 배치 크기, 에포크 수, ...)
- __model__: `keras.models.Model` 인스턴스.
    훈련 대상 모델에 대한 참조.

콜백 메소드들이 인자로 받는 `logs` 딕셔너리에
현재 배치 내지 에포크에 해당하는 수치들에 대한
키들이 담기게 된다.

현재 `Sequential` 모델 클래스의 `.fit()`
메소드에서 그 콜백들로 전달하는 `logs`에
다음 수치들이 들어 있게 된다.

- `on_epoch_end`: `logs`에 `acc`와 `loss`가 포함되며,
선택적으로 (`fit`에서 검증이 켜져 있으면)`val_loss`와
(검증과 정확도 감시가 켜져 있으면) `val_acc`가 포함된다.
- `on_batch_begin`: `logs`에 현재 패치의 표본 개수인
`size`가 들어간다.
- `on_batch_end`: `logs`에 `loss`가 포함되며,
선택적으로 (정확도 감시가 켜져 있으면) `acc`가 포함된다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L202)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger(stateful_metrics=None)
```

지표들의 에포크 평균값을 모으는 콜백.

모든 케라스 모델에 이 콜백이 자동으로 적용된다.

__인자__

- __stateful_metrics__: 한 에포크에서 평균하지 *말아야 할*
    지표들의 문자열 이름의 iterable.
    `on_epoch_end`에서 이 목록의 지표들은 원래 값 그대로 기록한다.
    `on_epoch_end`에서 나머지는 모두 평균을 내게 된다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L249)</span>
### TerminateOnNaN

```python
keras.callbacks.TerminateOnNaN()
```

NaN인 손실을 만났을 때 훈련을 종료시키는 콜백.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L262)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
```

stdout으로 지표를 찍는 콜백.

__인자__

- __count_mode__: "steps" 또는 "samples".
    진행 막대가 처리 단계(배치)와
    처리 표본 중 어느 쪽을 기준으로 하는지이다.
- __stateful_metrics__: 한 에포크에서 평균하지 *말아야 할*
    지표들의 문자열 이름의 iterable.
    이 목록의 지표들은 원래 값 그대로 기록한다.
    나머지(가령 손실 등)는 모두 평균을 내게 된다.

__예외__

- __ValueError__: `count_mode`가 유효하지 않은 경우.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L339)</span>
### History

```python
keras.callbacks.History()
```

`History` 객체에 이벤트들을 기록하는 콜백.

모든 케라스 모델에 이 콜백이 자동으로
적용된다. 그 `History` 객체가 모델의
`fit` 메소드에 의해 반환된다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L358)</span>
### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

각 에포크 후에 모델을 저장.

`filepath`에 이름 지정 형식 옵션이 있을 수 있어서
`epoch` 및 (`on_epoch_end`에 전달되는) `logs`의
키들의 값으로 채워진다.

예: `filepath`가 `weights.{epoch:02d}-{val_loss:.2f}.hdf5`이면
에포크 번호와 검증 손실이 들어간 파일명으로
모델 체크포인트가 저장된다.

__인자__

- __filepath__: str. 모델을 저장할 파일의 경로.
- __monitor__: 감시할 변량.
- __verbose__: 상세 출력 모드. 0 또는 1.
- __save_best_only__: `save_best_only=True`이면
    감시하는 변량 기준으로 가장 최근의 최고 모델을
    덮어 쓰지 않게 된다.
- __mode__: {auto, min, max} 중 하나.
    `save_best_only=True`이면 감시 변량의
    최대화 내지 최소화에 기반해서
    현재 저장 파일을 덮어 쓸지 여부를 결정한다.
    `val_acc`에서는 `max`여야 할 것이고,
    `val_loss`에서는 `min`이어야 하는 식이다.
    `auto` 모드에서는 감시 변량의 이름으로
    그 방향을 자동 추론한다.
- __save_weights_only__: True이면 모델의 가중치만 저장하고
    (`model.save_weights(filepath)`), 아니면 모델 전체를
    저장한다 (`model.save(filepath)`).
- __period__: 체크포인트 간격 (에포크 수).

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L458)</span>
### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)
```

감시하는 변량이 더는 개선되지 않을 때 훈련을 중단.

__인자__

- __monitor__: 감시할 변량.
- __min_delta__: 감시 변량에서 개선으로 인정할
    최소 변화량. 즉 절대적 변화량이 min_delta보다
    작으면 개선이 없다고 여기게 된다.
- __patience__: 개선 없이 에포크가 몇 번
    지나고 나면 훈련을 중단할 것인지.
- __verbose__: 상세 출력 모드.
- __mode__: {auto, min, max} 중 하나. `min`
    모드에서는 감시 변량이 감소하는 게
    멈췄을 때 훈련을 중단함. `max` 모드에서는
    감시 변량이 증가하는 게 멈췄을 때 중단함.
    `auto` 모드에서는 감시 변량 이름으로
    그 방향을 자동 추론함.
- __baseline__: 감시 변량이 도달해야 할 기준치.
    모델이 그 기준 너머로 개선되지 못하면
    훈련이 멈춘다.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L552)</span>
### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)
```

이벤트를 서버로 보내는 데 쓰는 콜백.

`requests` 라이브러리 필요함.
기본적으로 `root + '/publish/epoch/end/'`로 이벤트를 보낸다.
HTTP POST 방식 호출이며 `data` 인자는
이벤트 데이터를 JSON 인코딩 한 딕셔너리이다.
send_as_json이 True로 설정돼 있으면 요청의 컨텐츠 타입이 application/json이 된다.
그렇지 않으면 폼 한 개 안에 JSON들을 직렬화 해서 보낸다.

__인자__

- __root__: 문자열. 대상 서버의 루트 URL.
- __path__: 문자열. 이벤트를 보낼 `root` 기존 상대 경로.
- __field__: 문자열. 데이터를 저장하게 될 JSON 필드. 페이로드를 폼 한 개 내에서 보낼 때만
    (즉 send_as_json이 False로 설정돼 있을 때만) 그 필드를 쓴다.
- __headers__: 딕셔너리. 선택적인 커스텀 HTTP 헤더들.
- __send_as_json__: 불리언. 요청을 application/json으로 보낼지 여부.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L609)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

학습률 스케줄러.

__인자__

- __schedule__: 에포크 인덱스(int, 0부터 시작)와
    현재 학습률을 입력으로 받아서 새 학습률(float)을
    반환하는 함수.
- __verbose__: int. 0: 조용히, 1: 갱신 메시지.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L641)</span>
### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
```

텐서보드 기본 시각화.

[텐서보드](https://www.tensorflow.org/get_started/summaries_and_tensorboard)는
텐서플로우에서 제공하는 시각화 도구이다.

이 콜백은 텐서보드를 위한 로그를 기록한다.
그래서 훈련 및 테스트 지표들의 동적 그래프뿐 아니라
모델 각 층들에 대한 활성 히스토그램을 그릴 수 있게 된다.

텐서플로우를 pip로 설치했다면 명령행에서 다음처럼 해서
텐서보드를 띄울 수 있을 것이다.
```sh
tensorboard --logdir=/full_path_to_your_logs
```

텐서플로우 아닌 백엔드를 쓸 때에도 (텐서플라우를 설치했다면)
텐서보드가 동작하기는 할 것이다. 하지만 손실 및 지표 그래프를
표시하는 기능만 쓸 수 있을 것이다.

__인자__

- __log_dir__: 텐서보드에서 파싱 하게 될 로그 파일들을 저장할
    디렉터리의 경로.
- __histogram_freq__: 모델 층들의 활성 및 가중치 히스토그램을
    계산할 (에포크 기준) 빈도. 0으로 설정돼 있으면
    히스토그램을 계산하지 않는다. 히스토그램 시각화를 위해선
    검증 데이터(또는 검증 데이터 비율)을 지정해야 한다.
- __write_graph__: 텐서보드에서 그래프를 시각화할지 여부.
    write_graph를 True로 설정하면 로그 파일이 꽤 커질 수 있다.
- __write_grads__: 텐서보드에서 경사 히스토그램을 보일지 여부.
    `histogram_freq`가 0보다 커야 한다.
- __batch_size__: 히스토그램 계산을 위해 망에 넣어 주는
    입력 배치의 크기.
- __write_images__: 텐서보드에서 이미지로 시각화하도록
    모델 가중치를 기록할지 여부.
- __embeddings_freq__: 지정한 embedding 층들을 저장할
    (에코프 기준) 빈도. 0으로 설정돼 있으면 embedding을 계산하지 않는다.
    텐서보드의 Embedding 탭에서 시각화할 데이터는
    `embeddings_data`로 전달해야 한다.
- __embeddings_layer_names__: 주시할 층들의 이름 목록.
    None이거나 빈 리스트이면 모든 embedding 층을 감시하게 된다.
- __embeddings_metadata__: 층 이름을 그 embedding 층에 대한
    메타데이터가 저장될 파일 이름으로 매핑 하는 딕셔너리.
    메타데이터 파일 형식에 대해선
    [상세 설명](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional) 참고.
    모든 embedding 층에 같은 메타데이터 파일을 쓰는 경우에는
    문자열을 줄 수 있다.
- __embeddings_data__: `embeddings_layer_names`에 지정한 층들에
    embed 될 데이터. Numpy 배열이거나 (모델 입력이 하나인 경우),
    Numpy 배열들의 리스트 (모델 입력이 여럿인 경우).
    [embedding에 대해 더 배우기](https://www.tensorflow.org/programmers_guide/embedding).

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L951)</span>
### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
```

지표가 개선을 멈췄을 때 학습률을 줄인다.

모델 학습이 정체되었을 때 학습률을 2~10배로 줄이는 게
도움이 될 때가 많다. 이 콜백은 변량을 감시해서 에포크
'patience' 번에서 개선이 없으면 학습률을 줄인다.

__예시__


```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

__인자__

- __monitor__: 감시할 변량.
- __factor__: 학습률 감소 인자.
    new_lr = lr * factor
- __patience__: 개선 없이 에포크가 몇 번
    지나고 나면 학습률을 줄이게 되는지.
- __verbose__: int. 0: 조용히, 1: 갱신 메시지.
- __mode__: {auto, min, max} 중 하나. `min`
    모드에서는 감시 변량이 감소하는 게
    멈췄을 때 학습률을 줄임. `max`에서는
    감시 변량이 증가하는 게 멈췄을 때 줄임.
    `auto` 모드에서는 감시 변량 이름으로
    그 방향을 자동 추론함.
- __min_delta__: 새로운 최적이라고 판단하기 위한 문턱값.
    유의미한 변화에만 집중하기 위한 인자.
- __cooldown__: lr을 줄인 후에 이 에포크 수만큼
    지난 다음 정상 동작을 재개한다.
- __min_lr__: 학습률에 대한 하한.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1072)</span>
### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

에포크별 결과를 csv 파일로 보내는 콜백.

np.ndarray 같은 1차원 iterable을 포함해
문자열로 표현 가능한 모든 값들을 지원한다.

__예시__


```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```

__인자__

- __filename__: csv 파일의 이름. 가령 'run.log.csv'.
- __separator__: csv 파일에서 항목 구분에 쓸 문자열.
- __append__: True: 파일이 존재하면 덧붙이기 (훈련을
    재개할 때 유용함). False: 기존 파일을 덮어 쓰기.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1149)</span>
### LambdaCallback

```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

간단한 커스텀 콜백을 바로 만들 수 있는 콜백.

익명 함수들로 이 콜백을 구성하면 적절한 시점에 그 함수들이
호출된다. 참고로 콜백들은 다음처럼 위치가 정해진 인자들을 기대한다.

- `on_epoch_begin` 및 `on_epoch_end`는 위치 고정 인자 두 개 기대:
`epoch`, `logs`
- `on_batch_begin` 및 `on_batch_end`는 위치 고정 인자 두 개 기대:
`batch`, `logs`
- `on_train_begin` 및 `on_train_end`는 위치 고정 인자 한 개 기대:
`logs`

__인자__

- __on_epoch_begin__: 각 에포크 시작 시 호출.
- __on_epoch_end__: 각 에포크 끝에서 호출.
- __on_batch_begin__: 각 배치 시작 시 호출.
- __on_batch_end__: 각 배치 끝에서 호출.
- __on_train_begin__: 모델 훈련 시작 시 호출.
- __on_train_end__: 모델 훈련 끝에서 호출.

__예시__


```python
# 각 배치 시작에서 배치 번호 찍기
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# 에포크 손실을 JSON 형식 파일로 보낸다. 파일 내용이 제대로 된
# 형식의 JSON인 게 아니라 한 행에 JSON 객체가 하나씩 들어간다.
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

# 모델 훈련을 마친 후에 어떤 프로세스 끝내기.
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
        p.terminate() for p in processes if p.is_alive()])

model.fit(...,
          callbacks=[batch_print_callback,
                     json_logging_callback,
                     cleanup_callback])
```


---


# 콜백 만들기

기반 클래스 `keras.callbacks.Callback`을 확장해서 커스텀 콜백을 만들 수 있다. 콜백에서 클래스 속성 `self.model`을 통해 연계된 모델에 접근할 수 있다.

다음은 훈련 중 각 배치에 대한 손실 리스트를 저장하는 간단한 예시이다.
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### 예시: 손실 이력 기록하기

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)
# 출력
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### 예시: 모델 체크포인트

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
각 에포크 후에 검증 손실이 감소했으면 모델 가중치를 저장
'''
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
```
