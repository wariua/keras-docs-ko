<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L57)</span>
### MaxPooling1D

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```

시간 데이터를 위한 최대 풀링 연산.

__인자__

- __pool_size__: 정수. 최대 풀링 윈도의 크기.
- __strides__: 정수, 또는 None. 다운스케일 비율.
    가령 2로 하면 입력을 반으로 줄임.
    None이면 `pool_size` 사용.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).

__입력 형태__

`(batch_size, steps, features)` 형태의 3차원 텐서.

__출력 형태__

`(batch_size, downsampled_steps, features)` 형태의 3차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L169)</span>
### MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

공간 데이터를 위한 최대 풀링 연산.

__인자__

- __pool_size__: 정수 또는 정수 2개로 된 튜플.
    다운스케일 비율 (수직, 수평).
    (2, 2)라고 하면 두 공간 차원 모두에서 입력을 반으로 줄임.
    정수 하나만 지정하면 두 차원에 같은 윈도 길이를 쓰게 됨.
- __strides__: 정수, 또는 정수 2개로 된 튜플, 또는 None.
    보폭 값.
    None이면 `pool_size` 사용.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `channels_first`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4차원 텐서.

__출력 형태__

- `data_format='channels_last'`이면
    `(batch_size, pooled_rows, pooled_cols, channels)`
    형태의 4차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, pooled_rows, pooled_cols)`
    형태의 4차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L338)</span>
### MaxPooling3D

```python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

3차원 (공간 또는 공간-시간) 데이터를 위한 최대 풀링 연산.

__인자__

- __pool_size__: 정수 3개로 된 튜플.
    다운스케일 비율 (dim1, dim2, dim3로).
    (2, 2, 2)라고 하면 3차원 입력의 각 차원에서 크기를 반으로 줄임.
- __strides__: 정수 3개로 된 튜플, 또는 None. 보폭 값.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 입력에 해당하고
    `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5차원 텐서.

__출력 형태__

- `data_format='channels_last'`이면
    `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
    형태의 5차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
    형태의 5차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L87)</span>
### AveragePooling1D

```python
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid')
```

시간 데이터를 위한 평균 최대 연산.

__인자__

- __pool_size__: 정수. 평균 풀링 윈도의 크기.
- __strides__: 정수, 또는 None. 다운스케일 비율.
    가령 2로 하면 입력을 반으로 줄임.
    None이면 `pool_size` 사용.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).

__입력 형태__

`(batch_size, steps, features)` 형태의 3차원 텐서.

__출력 형태__

`(batch_size, downsampled_steps, features)` 형태의 3차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L224)</span>
### AveragePooling2D

```python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

공간 데이터를 위한 평균 풀링 연산.

__인자__

- __pool_size__: 정수 또는 정수 2개로 된 튜플.
    다운스케일 비율 (수직, 수평).
    (2, 2)라고 하면 두 공간 차원 모두에서 입력을 반으로 줄임.
    정수 하나만 지정하면 두 차원에 같은 윈도 길이를 쓰게 됨.
- __strides__: 정수, 또는 정수 2개로 된 튜플, 또는 None.
    보폭 값.
    None이면 `pool_size` 사용.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `channels_first`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4차원 텐서.

__출력 형태__

- `data_format='channels_last'`이면
    `(batch_size, pooled_rows, pooled_cols, channels)`
    형태의 4차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, pooled_rows, pooled_cols)`
    형태의 4차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L388)</span>
### AveragePooling3D

```python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

3차원 (공간 또는 공간-시간) 데이터를 위한 평균 풀링 연산.

__인자__

- __pool_size__: 정수 3개로 된 튜플.
    다운스케일 비율 (dim1, dim2, dim3로).
    (2, 2, 2)라고 하면 3차원 입력의 각 차원에서 크기를 반으로 줄임.
- __strides__: 정수 3개로 된 튜플, 또는 None. 보폭 값.
- __padding__: `"valid"` 또는 `"same"` (대소문자 구분 없음).
- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 입력에 해당하고
    `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5차원 텐서.

__출력 형태__

- `data_format='channels_last'`이면
    `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
    형태의 5차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
    형태의 5차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L469)</span>
### GlobalMaxPooling1D

```python
keras.layers.GlobalMaxPooling1D()
```

시간 데이터를 위한 전역 최대 풀링 연산.

__입력 형태__

`(batch_size, steps, features)` 형태의 3차원 텐서.

__출력 형태__

`(batch_size, features)` 형태의 2차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L454)</span>
### GlobalAveragePooling1D

```python
keras.layers.GlobalAveragePooling1D()
```

시간 데이터를 위한 전역 평균 풀링 연산.

__입력 형태__

`(batch_size, steps, features)` 형태의 3차원 텐서.

__출력 형태__

`(batch_size, features)` 형태의 2차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L544)</span>
### GlobalMaxPooling2D

```python
keras.layers.GlobalMaxPooling2D(data_format=None)
```

공간 데이터를 위한 전역 최대 풀링 연산.

__인자__

- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `channels_first`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4차원 텐서.

__출력 형태__

`(batch_size, channels)` 형태의 2차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L509)</span>
### GlobalAveragePooling2D

```python
keras.layers.GlobalAveragePooling2D(data_format=None)
```

공간 데이터를 위한 전역 평균 풀링 연산.

__인자__

- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는 `(batch, height, width, channels)`
    형태의 입력에 해당하고
    `channels_first`는 `(batch, channels, height, width)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format='channels_last'`이면
    `(batch_size, rows, cols, channels)`
    형태의 4차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, rows, cols)`
    형태의 4차원 텐서.

__출력 형태__

`(batch_size, channels)` 형태의 2차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L639)</span>
### GlobalMaxPooling3D

```python
keras.layers.GlobalMaxPooling3D(data_format=None)
```

3차원 데이터를 위한 전역 최대 풀링 연산.

__인자__

- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 입력에 해당하고
    `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5차원 텐서.

__출력 형태__

`(batch_size, channels)` 형태의 2차원 텐서.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L604)</span>
### GlobalAveragePooling3D

```python
keras.layers.GlobalAveragePooling3D(data_format=None)
```

3차원 데이터를 위한 전역 평균 풀링 연산.

__인자__

- __data_format__: 문자열.
    `channels_last`(기본값) 또는 `channels_first`.
    입력에서 차원들의 순서.
    `channels_last`는
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 입력에 해당하고
    `channels_first`는
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 입력에 해당한다.
    지정하지 않으면 케라스 설정 파일 `~/.keras/keras.json`에
    있는 `image_data_format` 값을 쓴다.
    그 값을 설정한 적이 없으면 "channels_last"를 쓰게 된다.

__입력 형태__

- `data_format='channels_last'`이면
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    형태의 5차원 텐서.
- `data_format='channels_first'`이면
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`
    형태의 5차원 텐서.

__출력 형태__

`(batch_size, channels)` 형태의 2차원 텐서.
