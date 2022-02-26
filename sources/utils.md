<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/utils/generic_utils.py#L21)</span>
### CustomObjectScope

```python
keras.utils.CustomObjectScope()
```

`_GLOBAL_CUSTOM_OBJECTS`에 대한 변경 내용이 빠져나가지 못하는 스코프 만들기

`with` 문 내의 코드에서 커스텀 객체를 이름으로 접근할 수 있게 된다.
전역 커스텀 객체 세트에 대한 변경 사항이 `with` 문으로 감싼 범위 내에서
유지된다. `with` 문이 끝나면 전역 커스텀 객체 세트가 `with` 문
시작 시점의 상태로 되돌아간다.

__예시__


커스텀 객체 `MyObject`(가령 어떤 클래스)가 있을 때,

```python
with CustomObjectScope({'MyObject':MyObject}):
    layer = Dense(..., kernel_regularizer='MyObject')
    # save, load 등에서 커스텀 객체를 이름으로 인식하게 됨
```

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/utils/io_utils.py#L16)</span>
### HDF5Matrix

```python
keras.utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

Numpy 배열 대신 사용할 HDF5 데이터셋 표현.

__예시__


```python
x_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(x_data)
```

`start`와 `end`를 주면 데이터셋 일부만 쓸 수 있다.

선택적으로 정규화 함수(또는 람다)를 줄 수 있다. 가져오는 데이터의
조각마다 호출된다.

__인자__

- __datapath__: 문자열. HDF5 파일 경로.
- __dataset__: 문자열. datapath에 지정한 파일 내의 HDF5 데이터셋
    이름.
- __start__: int. 지정한 데이터셋에서 원하는 부분 시작점.
- __end__: int. 지정한 데이터셋에서 원하는 부분 끝점.
- __normalizer__: 가져온 데이터마다 호출할 함수.

__반환__

배열 같은 HDF5 데이터셋.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302)</span>
### Sequence

```python
keras.utils.Sequence()
```

데이터셋 같은 데이터 열을 나눠 제공하기 위한 기반 클래스.

모든 `Sequence`는 `__getitem__` 메소드와 `__len__` 메소드를 구현해야 한다.
에포크 사이에서 데이터셋을 변경하고 싶다면 `on_epoch_end`를 구현할 수 있다.
`__getitem__` 메소드에선 꽉 채운 배치를 반환하는 게 좋다.

__주의__


다중 프로세싱을 더 안전하게 하는 방법이 `Sequence`다. 이 구조를 쓰면 망에서
에포크마다 각 샘플을 한 번씩 훈련에 쓰는 게 보장된다. 제너레이터에선 그렇지 않다.

__예시__


```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# 여기서 `x_set`은 이미지 경로들의 리스트이고
# `y_set`은 연계된 클래스들이다.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)
```

----

### to_categorical


```python
keras.utils.to_categorical(y, num_classes=None)
```


클래스 벡터(정수들)를 이진 클래스 행렬로 변환.

그 결과를 categorical_crossentropy 등에 쓸 수 있다.

__인자__

- __y__: 행렬로 변환할 클래스 벡터. (0에서 num_classes 사이 정수들)
- __num_classes__: 클래스 수.

__반환__

입력의 이진 행렬 표현. 클래스 축이 마지막에 위치한다.

----

### normalize


```python
keras.utils.normalize(x, axis=-1, order=2)
```


Numpy 배열 정규화.

__인자__

- __x__: 정규화할 Numpy 배열.
- __axis__: 정규화할 축.
- __order__: 정규화 차수. (가령 L2 norm이면 2)

__반환__

정규화된 배열 사본.

----

### get_file


```python
keras.utils.get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
```


캐시에 없으면 URL에서 파일을 내려받기.

기본적으로 url `origin`의 파일을 내려받아서 cache_dir `~/.keras`에
있는 cache_subdir `datasets`에 `fname`이라는 파일 이름으로 저장한다.
가령 `example.txt` 파일의 최종 위치는 `~/.keras/datasets/example.txt`가
된다.

또한 tar, tar.gz, tar.bz, zip 형식 파일 압축을 풀 수 있다. 해시를
주면 내려받은 파일을 검사한다. 명령행 프로그램 `shasum` 및
`sha256sum`으로 해시를 계산할 수 있다.

__인자__

- __fname__: 파일 이름. 절대 경로 `/path/to/file.txt`를 지정하면
    그 위치에 파일이 저장된다.
- __origin__: 파일 원천 URL.
- __untar__: 'extract'로 대체됨.
    불리언. 파일을 압축해야 할지 여부.
- __md5_hash__: 'file_hash'로 대체됨.
    검사에 쓸 파일의 md5 해시.
- __file_hash__: 내려받은 파일에 기대하는 해시 문자열.
    해시 알고리즘 sha256과 md5 모두를 지원한다.
- __cache_subdir__: 파일을 저장할 케라스 캐시 디렉터리 아래
    서브디렉터리. 정대 경로 `/path/to/folder`를 지정하면
    그 위치에 파일이 저장된다.
- __hash_algorithm__: 파일 검사에 쓸 해시 알고리즘 선택.
    선택지는 'md5', 'sha256', 'auto'다.
    기본값 'auto'는 사용 중인 해시 알고리즘을 자동 탐지한다.
- __extract__: 참이면 tar나 zip 같은 아카이브 파일 압축 해제 시도.
- __archive_format__: 파일 압축을 풀어 볼 아카이브 형식.
    선택지는 'auto', 'tar', 'zip', None.
    'tar'는 tar, tar.gz, tar.bz를 포함한다.
    기본값 'auto'는 ['tar', 'zip']이다.
    None이나 빈 리스트면 반환하는 일치 항목이 없게 된다.
- __cache_dir__: 파일 캐시를 저장할 위치. None이면 기본값으로
    [케라스 디렉터리](/getting-started/faq/#_19)를 쓴다.

__반환__

내려받은 파일의 경로.

----

### print_summary


```python
keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)
```


모델 개요 찍기.

__인자__

- __model__: 케라스 모델 인스턴스.
- __line_length__: 행 출력 길이.
    (가령 다양한 터미널 창 크기에 맞춰 표시하도록 이 값을
    설정할 수 있다.)
- __positions__: 각 행에서 로그 항목들의 상대적 또는 절대적 위치들.
    주지 않으면 기본은 `[.33, .55, .67, 1.]`.
- __print_fn__: 사용할 출력 함수.
    개요 행마다 이 함수가 호출된다.
    자체 함수를 설정해서 개요를 문자열 형태로
    잡아낼 수 있다.
    기본은 `print`(stdout으로 출력)이다.

----

### plot_model


```python
keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
```


케라스 모델을 dot 형식으로 변환해서 파일에 저장.

__인자__

- __model__: 케라스 모델 인스턴스.
- __to_file__: plot 이미지 파일 이름.
- __show_shapes__: 형태 정보 표시할지 여부.
- __show_layer_names__: 층 이름 표시할지 여부.
- __rankdir__: PyDot에 주는 `rankdir` 인자이며
    도표 형식을 지정하는 문자열이다.
    'TB'는 수직 방향 도표를 만들고
    'LR'은 수평 방향 도표를 만든다.

----

### multi_gpu_model


```python
keras.utils.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False)
```


모델을 여러 GPU로 복제.

구체적으로 이 함수는 단일 머신 다중 GPU 데이터 병렬
처리를 구현한다. 다음처럼 동작한다.

- 모델의 입력(들)을 여러 배치 조각으로 나눈다.
- 각 배치 조각에 모델 사본을 적용한다. 각 모델 사본은
별도 GPU 상에서 실행된다.
- 결과물을 (CPU에서) 큰 배치 하나로 합친다.

가령 `batch_size`가 64인데 `gpus=2`를 쓰면 입력을 32개
표본으로 된 배치 조각 2개로 나누고, 각 GPU에서
배치 조각 하나씩을 처리하고, 처리한 64개 표본으로 된
전체 배치를 반환한다.

8개 GPU까지 준선형으로 속도가 올라간다.

현재 이 함수는 텐서플로우 백엔드에만 사용 가능하다.

__인자__

- __model__: 케라스 모델 인스턴스. OOM 오류를 피하기 위해
    가령 이 모델을 CPU 상에 구성할 수도 있을 것이다.
    (아래의 사용례 참고.)
- __gpus__: 2 이상의 정수 또는 정수들의 리스트. GPU 개수
    또는 모델 사본을 생성할 GPU들의 ID 목록.
- __cpu_merge__: 모델 가중치 병합을 CPU 영역에서 하도록
    강제할지 여부를 나타내는 불리언 값.
- __cpu_relocation__: 모델의 가중치를 CPU 영역에서 생성할지
    여부를 나타내는 불리언 값. 앞선 장치 영역들에서 모델에
    정의되지 않더라도 이 옵션을 켜서 대처할 수 있다.

__반환__

케라스`Model` 인스턴스. 첫 번째 인자 `model`처럼 사용할 수 있으면서도
작업을 여러 GPU로 분산시켜 준다.

__예시 1 - CPU에서 가중치 병합 방식으로 모델 훈련시키기__

$Example_2_-_Training_models_with_weights_merge_on_CPU_using_cpu_relocation$0

__예시 2 - CPU에서 가중치 병합 방식으로 cpu_relocation 써서 모델 훈련시키기__

$Example_2_-_Training_models_with_weights_merge_on_CPU_using_cpu_relocation$1

__예시 3 - GPU에서 가중치 병합 방식으로 모델 훈련시키기 (NV-링크 사용 시 권장)__

$Example_2_-_Training_models_with_weights_merge_on_CPU_using_cpu_relocation$2

__모델 저장에 대해__


다중 GPU 모델을 저장할 때는 `multi_gpu_model`이 반환한 모델이 아니라
템플릿 모델(`multi_gpu_model`에 준 인자)을 `.save(fname)` 내지
`.save_weights(fname)`에 사용해야 한다.
