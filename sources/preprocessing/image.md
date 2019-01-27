
# 이미지 전처리

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L527)</span>
## ImageDataGenerator 클래스

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0)
```

실시간 데이터 증강으로 텐서 이미지 데이터 배치를 생성한다.
데이터 상에서 (배치 단위로) 루프를 돈다.

__인자__

- __featurewise_center__: bool.
    데이터셋에서 입력 평균을 0으로 만들기. 피쳐 단위.
- __samplewise_center__: bool. 각 표본 평균을 0으로 만들기.
- __featurewise_std_normalization__: bool.
    데이터셋 표준 편차로 입력을 나누기. 피쳐 단위.
- __samplewise_std_normalization__: bool. 각 입력을 표준 편차로 나누기.
- __zca_epsilon__: ZCA 백화를 위한 엡실론 값. 기본값은 1e-6.
- __zca_whitening__: bool. ZCA 백화 적용 여부.
- __rotation_range__: int. 무작위 회전 각도 범위.
- __width_shift_range__: float, 또는 1차원 배열, 또는 int.
    - float: 1보다 작으면 전체 폭에서의 비율, 아니면 픽셀 수.
    - 1차원 배열: 배열에서 무작위로 뽑은 항목들.
    - int: `(-width_shift_range, +width_shift_range)` 구간에서
        뽑은 정수 픽셀 수.
    - `width_shift_range=2`라고 하면 가능한 값이
        정수 `[-1, 0, +1]`이며,
        `width_shift_range=[-1, 0, +1]`과 동일함.
        반면 `width_shift_range=1.0`이라고 하면 가능한 값이
        [-1.0, +1.0) 구간의 실수들임.
- __height_shift_range__: float, 또는 1차원 배열, 또는 int.
    - float: 1보다 작으면 전체 높이에서의 비율, 아니면 픽셀 수.
    - 1차원 배열: 배열에서 무작위로 뽑은 항목들.
    - int: `(-height_shift_range, +height_shift_range)` 구간에서
        뽑은 정수 픽셀 수.
    - `height_shift_range=2`라고 하면 가능한 값이
        정수 `[-1, 0, +1]`이며,
        `height_shift_range=[-1, 0, +1]`과 동일함.
        반면 `height_shift_range=1.0`이라고 하면 가능한 값이
        [-1.0, +1.0) 구간의 실수들임.
- __brightness_range__: float 두 개짜리 튜플 내지 리스트.
    명도 변화 정도를 고르는 범위.
- __shear_range__: float. 전단 변환 정도.
    (반시계 방향 도 단위의 전단 각.)
- __zoom_range__: float 또는 [lower, upper]. 무작위 확대 범위.
    float인 경우 `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
- __channel_shift_range__: float. 무작위 채널 이동 범위.
- __fill_mode__: {"constant", "nearest", "reflect" or "wrap"} 중 하나.
    기본값은 'nearest'.
    입력 가장자리 밖의 점들을 지정한 모드에 따라 채운다.
    - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
    - 'nearest':  aaaaaaaa|abcd|dddddddd
    - 'reflect':  abcddcba|abcd|dcbaabcd
    - 'wrap':  abcdabcd|abcd|abcdabcd
- __cval__: float 또는 int.
    `fill_mode = "constant"`일 때 가장자리 밖의
    점을 채우는 값.
- __horizontal_flip__: bool. 무작위로 입력을 좌우로 뒤집기.
- __vertical_flip__: bool. 무작위로 입력을 상하로 뒤집기.
- __rescale__: 스케일 계수. 기본값은 None.
    None이나 0이면 스케일 조정을 하지 않고,
    아니면 지정한 값을 (다른 변환들을 모두 적용한 후에)
    데이터에 곱한다.
- __preprocessing_function__: 각 입력에 적용할 함수.
    이미지 크기 조정 및 증강 후에 이 함수를 실행한다.
    이 함수는 이미지 하나(랭크 3짜리 Numpy 텐서)를
    인자로 받아서 같은 형태의 Numpy 텐서를 출력해야 한다.
- __data_format__: 이미지 데이터 형식.
    "channels_first" 또는 "channels_last".
    "channels_last" 모드는 이미지가
    `(samples, height, width, channels)` 형태라는 것이고
    "channels_first" 모드는 이미지가
    `(samples, channels, height, width)` 형태라는 것이다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.
- __validation_split__: float. 검증을 위해 떼어놓을 이미지들의 비율.
    (양끝 뺀 0과 1 사이.)

__예시__

`.flow(x, y)` 사용 예:

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# 피쳐 단위 정규화에 필요한 수치들(표준편차, 평균, ZCA 백화
# 적용 시 주성분) 계산
datagen.fit(x_train)

# 실시간 데이터 증강 한 배치들에 모델 맞추기
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# 좀 더 "수동"으로
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # 제너레이터가 루프를 무한히 돌기 때문에
            # 직접 루프에서 나가야 한다
            break
```
`.flow_from_directory(directory)` 사용 예:

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

이미지와 마스크를 함께 변환하는 예:

```python
# 같은 인자로 인스턴스를 두 개 만든다
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# fit 메소드와 flow 메소드에 같은 시드와 키워드 인자 주기
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# 이미지와 마스크를 내놓는 제너레이터 한 개로 합치기
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```


---
## ImageDataGenerator 메소드

### apply_transform


```python
apply_transform(x, transform_parameters)
```


주어진 매개변수들에 따라 이미지에 변환을 적용한다.

__인자__

- __x__: 3차원 텐서. 단일 이미지.
- __transform_parameters__: 변형을 기술하는 문자열-매개변수
    쌍들의 딕셔너리.
    현재 딕셔너리의 다음 매개변수들을
    사용한다.
    - `'theta'`: float. 도 단위 회전 각도.
    - `'tx'`: float. x 방향 이동.
    - `'ty'`: float. y 방향 이동.
    - `'shear'`: float. 도 단위 전단 각도.
    - `'zx'`: float. x 방향 확대 비율.
    - `'zy'`: float. y 방향 확대 비율.
    - `'flip_horizontal'`: bool. 좌우로 뒤집기.
    - `'flip_vertical'`: bool. 상하로 뒤집기.
    - `'channel_shift_intencity'`: float. 채널 이동 정도.
    - `'brightness'`: float. 명암 이동 정도.

__반환__

입력을 변환한 버전 (같은 형태).

---
### fit


```python
fit(x, augment=False, rounds=1, seed=None)
```


어떤 표본 데이터에 데이터 제너레이터를 맞춘다.

표본 데이터 배열을 가지고 데이터 의존적 변형들에
관련된 내부 데이터 통계들을 계산한다.

`featurewise_center`나 `featurewise_std_normalization`,
`zca_whitening`이 True로 설정돼 있는 경우에만 필요하다.

__인자__

- __x__: 표본 데이터. 랭크가 4여야 함.
 그레이스케일 데이터인 경우
 채널 축 값이 1이어야 하고,
 RGB 데이터인 경우 값이 3이어야 하고,
 RGBA 데이터인 경우 값이 4여야 한다.
- __augment__: bool (기본값: False).
    무작위로 증강한 표본들에 맞출지 여부.
- __rounds__: int (기본값: 1).
    데이터 증강 사용 시 (`augment=True`)
    데이터에 몇 번의 증강을 돌려서 쓸 것인지.
- __seed__: int (기본값: None). 난수 시드.
   
---
### flow


```python
flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
```


데이터 배열과 레이블 배열을 받아서 증강 데이터 배치들을 생성한다.

__인자__

- __x__: 입력 데이터. 랭크 4인 numpy 배열 또는 튜플.
    튜플인 경우 첫 번째 항목에 이미지들이 있어야 하며
    두 번째 항목은 또 다른 numpy 배열이나 numpy 배열들의
    리스트이고 변경 없이 출력으로 전달된다.
    이를 이용해 이미지와 더불어 다양한 데이터를
    모델에 넣어 줄 수 있다.
    그레이스케일 데이터인 경우 이미지 배열의 채널 축
    값이 1이어야 하고, RGB 데이터인 경우 값이 3이어야 하고,
    RGBA 데이터인 경우 값이 4여야 한다.
- __y__: 레이블들.
- __batch_size__: int (기본값: 32).
- __shuffle__: bool (기본값: True).
- __sample_weight__: 표본 가중치.
- __seed__: int (기본값: None).
- __save_to_dir__: None 또는 str (기본값: None).
    선택적인 이 옵션을 이용해 생성되는 증강된
    그림들을 저장할 디렉터리를 지정할 수 있다.
    (뭘 하고 있는지 시각화 하는 데 유용하다.)
- __save_prefix__: str (기본값: `''`).
    저장하는 그림들의 파일명 앞에 붙일 문자열.
    (`save_to_dir`이 설정된 경우에만 의미 있음).
- __save_format__: "png" 또는 "jpeg"
    (`save_to_dir`이 설정된 경우에만 의미 있음). 기본값: "png".
- __subset__: `ImageDataGenerator`에 `validation_split`이 설정된
    경우에 데이터가 어느 쪽인지 (`"training"` 또는 `"validation"`).

__반환__

`(x, y)` 튜플들을 내놓는 `Iterator`.
    여기서 `x`는 (단일 이미지 입력인 경우)
    이미지 데이터의 numpy 배열이거나
    (입력이 더 있는 경우)
    numpy 배열들의 리스트이며
    `y`는 대응하는 레이블의 numpy 배열이다.
    'sample_weight'가 None이 아니면
    내놓는 튜플들이 `(x, y, sample_weight)` 형태이다.
    `y`가 None이면 numpy 배열 `x`만 반환한다.

---
### flow_from_directory


```python
flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')
```


디렉터리 경로를 받아서 증강 데이터 배치들을 생성한다.

__인자__

- __directory__: 대상 디렉터리 경로.
    클래스별로 서브디렉터리가 하나씩 있어야 한다.
    그 서브디렉터리 각각의 안에 있는 PNG, JPG, BMP, PPM, TIF
    이미지들이 모두 제너레이터에 들어가게 된다.
    자세한 내용은 [이 스크립트](
    https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
    참고.
- __target_size__: 정수 튜플 `(height, width)`,
    기본값: `(256, 256)`.
    찾은 이미지들을 모두 이 크기로 조정한다.
- __color_mode__: "grayscale", "rbg", "rgba" 중 하나. 기본값: "rgb".
    이미지를 채널 1개, 3개, 4개 중 어느 것으로 변환할 것인지.
- __classes__: 선택적. 클래스 서브디렉터리들의 리스트
    (가령 `['dogs', 'cats']`). 기본값: None.
    주지 않으면 `directory` 아래 서브디렉터리 이름/구조를
    가지고 클래스 목록을 자동으로 추론하게 된다.
    각 서브디렉터리를 별개의 클래스로 처리한다.
    (레이블 인덱스로 연결될 클래스들의 순서는
    알파벳 순서이다.)
    `class_indices` 속성을 통해 클래스 이름에서
    레이블 인덱스로 가는 매핑을 담은 딕셔너리를
    얻을 수 있다.
- __class_mode__: "categorical", "binary", "sparse", "input",
    None 중 하나. 기본값: "categorical".
    반환되는 레이블 배열의 타입을 결정한다.
    - "categorical"이면 2차원 원핫 인코딩 레이블.
    - "binary"이면 1차원 바이너리 레이블,
        "sparse"이면 1차원 정수 레이블.
    - "input"이면 입력 이미지가 동일한
        이미지 (오토인코더를 쓸 때 주로 유용함).
    - None이면 레이블이 반환되지 않는다.
      (제너레이터가 이미지 데이터들의 배치만
      내놓음. `model.predict_generator()`,
      `model.evaluate_generator()` 등에 쓰기에 유용함.)
      참고로 class_mode가 None인 경우에도
      `directory`의 서브디렉터리 안에
      데이터가 있어야 제대로 동작한다.
- __batch_size__: 데이터 배치 크기 (기본값: 32).
- __shuffle__: 데이터를 뒤섞을지 여부 (기본값: True)
- __seed__: 선택적. 뒤섞기 및 변형에 쓸 난수 시드.
- __save_to_dir__: None 또는 str (기본값: None).
    선택적인 이 옵션을 이용해 생성되는 증강된
    그림들을 저장할 디렉터리를 지정할 수 있다.
    (뭘 하고 있는지 시각화 하는 데 유용하다.)
- __save_prefix__: str. 저장하는 그림들의 파일명 앞에 붙일 문자열.
    (`save_to_dir`이 설정된 경우에만 의미 있음).
- __save_format__: "png" 또는 "jpeg"
    (`save_to_dir`이 설정된 경우에만 의미 있음). 기본값: "png".
- __follow_links__: 클래스 서브디렉터리들 안에서 심볼릭 링크를
    따라갈지 여부 (기본값: False).
- __subset__: `ImageDataGenerator`에 `validation_split`이 설정된
    경우에 데이터가 어느 쪽인지 (`"training"` 또는 `"validation"`).
- __interpolation__: 적재된 이미지 크기와 대상 크기가 다른 경우
    이미지 재배열에 사용할 보간법.
    `"nearest"`, `"bilinear"`, `"bicubic"`을 지원한다.
    PIL 버전 1.1.3이나 그 이상이 설치돼 있으면 `"lanczos"`도
    지원한다. PIL 버전 3.4.0이나 그 이상이 설치돼 있으면
    `"box"` 및 `"hamming"`도 지원한다.
    지정하지 않으면 `"nearest"`를 쓴다.

__반환__

`(x, y)` 튜플들을 내놓는 `DirectoryIterator`.
    여기서 `x`는 `(batch_size, *target_size, channels)` 형태
    이미지들의 배치를 담은 numpy 배열이며
    `y`는 대응하는 레이블의 numpy 배열이다.

---
### get_random_transform


```python
get_random_transform(img_shape, seed=None)
```


변환을 위한 무작위 매개변수를 생성한다.

__인자__

- __seed__: 난수 시드.
- __img_shape__: 정수 튜플.
    변환할 이미지의 형태.

__반환__

변환을 기술하는 무작위 선정 매개변수들을 담은 딕셔너리.

---
### random_transform


```python
random_transform(x, seed=None)
```


이미지에 무작위 변환을 적용한다.

__인자__

- __x__: 3차원 텐서. 단일 이미지.
- __seed__: 난수 시드.

__반환__

입력을 무작위로 변환한 버전 (같은 형태).

---
### standardize


```python
standardize(x)
```


입력들의 배치에 정규화 설정을 적용한다.

__인자__

- __x__: 정규화 할 입력 배치.

__반환__

입력을 정규화 한 결과.

