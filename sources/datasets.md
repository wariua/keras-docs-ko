# 데이터셋

## CIFAR10 소형 이미지 분류

10가지 분류로 레이블이 붙은 32x32 컬러 훈련 이미지 50,000개, 그리고 테스트 이미지 10,000개로 된 데이터셋.

### 사용법:

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

- __반환:__
    - 튜플 2개:
        - __x_train, x_test__: 백엔드 설정 `image_data_format`이 `channels_first`냐 `channels_last`냐에 따라 (num_samples, 3, 32, 32) 또는 (num_samples, 32, 32, 3) 형태인 RGB 이미지 데이터의 uint8 배열.
        - __y_train, y_test__: (num_samples) 형태인 분류 레이블(0-9 범위의 정수)의 uint8 배열.


---

## CIFAR100 소형 이미지 분류

100가지 분류로 레이블이 붙은 32x32 컬러 훈련 이미지 50,000개, 그리고 테스트 이미지 10,000개로 된 데이터셋.

### 사용법:

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```

- __반환:__
    - 튜플 2개:
        - __x_train, x_test__: 백엔드 설정 `image_data_format`이 `channels_first`냐 `channels_last`냐에 따라 (num_samples, 3, 32, 32) 또는 (num_samples, 32, 32, 3) 형태인 RGB 이미지 데이터의 uint8 배열.
        - __y_train, y_test__: (num_samples) 형태인 분류 레이블의 uint8 배열.

- __인자:__

    - __label_mode__: "fine" 또는 "coarse".


---

## IMDB 영화 감상 감상평 감정 분류

IMDB에서 가져온 영화 감상평 25,000개에 감정(긍정/부정) 레이블을 붙인 데이터셋. 감상평들을 전처리해서 각 감상평을 단어 인덱스(정수)들의 [열](preprocessing/sequence.md)로 인코딩 한다. 편의를 위해 데이터셋 전체에서의 빈도에 따라 단어들에 번호가 붙어 있다. 그래서 예를 들어 정수 "3"은 데이터에서 3번째로 자주 등장하는 단어를 인코딩 한다. 이를 이용하면 "가장 흔한 단어 상위 10,000개만 고려하되 가장 흔한 단어 상위 20개는 제외" 같은 필터링 동작을 빠르게 할 수 있다.

편의상 "0"은 특정 단어를 가리키는 게 아니라 알 수 없는 단어를 인코딩 하는 데 쓴다.

### 사용법:

```python
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
```
- __반환:__
    - 튜플 2개:
        - __x_train, x_test__: 인덱스(정수)들의 리스트인 열들의 리스트. num_words 인자를 지정했으면 가능한 인덱스 최댓값이 num_words-1이다. maxlen 인자를 지정했으면 가능한 최대 열 길이가 maxlen이다.
        - __y_train, y_test__: 정수 레이블(1 또는 0)들의 리스트.

- __인자:__

    - __path__: 로컬에 (`'~/.keras/datasets/' + path`에) 데이터가 없으면 이 위치에 내려받는다.
    - __num_words__: int 또는 None. 이보다 덜 빈번한 단어는 열 데이터에 `oov_char` 값으로 나오게 된다.
    - __skip_top__: int. 무시할 가장 빈번한 단어들. (열 데이터에 `oov_char` 값으로 나오게 된다.)
    - __maxlen__: int. 최대 열 길이. 더 긴 열은 잘리게 된다.
    - __seed__: int. 재현 가능한 데이터 뒤섞기를 위한 시드.
    - __start_char__: int. 열 시작점을 이 문자로 표시하게 된다.
        0은 일반적으로 패딩 문자이므로 1로 설정돼 있다.
    - __oov_char__: int. `num_words`나 `skip_top` 때문에 잘린
        단어들이 이 문자로 바뀌게 된다.
    - __index_from__: int. 실제 단어들에 이 인덱스부터 번호를 붙인다.


---

## 로이터 뉴스 서비스 주제 분류

로이터에서 가져온 뉴스 11,228개에 46가지 주제로 레이블을 붙인 데이터셋. IMDB 데이터셋에서처럼 각 뉴스가 단어 인덱스(같은 방식)의 열로 인코딩 돼 있다.

### 사용법:

```python
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```

IMDB 데이터셋과 내용이 같되 추가로 다음 인자가 있다.

- __test_split__: float. 데이터셋에서 테스트 데이터로 쓸 부분의 비율.

이 데이터셋에서는 열 인코딩에 쓰인 단어 인덱스를 얻을 수도 있다.

```python
word_index = reuters.get_word_index(path="reuters_word_index.json")
```

- __반환:__ 키가 단어(str)이고 값이 인덱스(int)인 딕셔너리. 가령 `word_index["giraffe"]`가 `1234`를 반환할 수 있을 것이다.

- __인자:__

    - __path__: 로컬에 (`'~/.keras/datasets/' + path`에) 인덱스 파일이 없으면 이 위치에 내려받는다.
    

---

## MNIST 필기 숫자 데이터베이스

10가지 숫자들의 28x28 그레이스케일 이미지 60,000개짜리 데이터셋과 이미지 10,000개짜리 테스트셋.

### 사용법:

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

- __반환:__
    - 튜플 2개:
        - __x_train, x_test__: (num_samples, 28, 28) 형태인 그레이스케일 이미지 데이터의 uint8 배열.
        - __y_train, y_test__: (num_samples) 형태인 숫자 레이블(0-9 범위의 정수)의 uint8 배열.

- __인자:__

    - __path__: 로컬에 (`'~/.keras/datasets/' + path`에) 인덱스 파일이 없으면 이 위치에 내려받는다.


---

## 패션 기사들의 패션-MNIST 데이터베이스

10가지 패션 분류의 28x28 그레이스케일 이미지 60,000개짜리 데이터셋과 이미지 10,000개짜리 테스트셋. 이 데이터셋을 그대로 MNIST 대신에 쓸 수 있다. 분류 레이블은 다음과 같다.

| 레이블 | 설명 |
| --- | --- |
| 0 | 티셔츠/상의 |
| 1 | 바지 |
| 2 | 풀오버 |
| 3 | 드레스 |
| 4 | 코트 |
| 5 | 샌들 |
| 6 | 셔츠 |
| 7 | 스니커 |
| 8 | 가방 |
| 9 | 발목 부츠 |

### 사용법:

```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

- __반환:__
    - 튜플 2개:
        - __x_train, x_test__: (num_samples, 28, 28) 형태인 그레이스케일 이미지 데이터의 uint8 배열.
        - __y_train, y_test__: (num_samples) 형태인 분류 레이블(0-9 범위의 정수)의 uint8 배열.


---

## 보스턴 주택 가격 회귀 데이터셋


카네기 멜론 대학교에서 유지하는 StatLib 라이브러리에서 가져 온 데이터셋이다.

1970년대 말 보스턴 교외 근방 여러 위치의 주택들의 13가지 속성을 담은 표본들이다.
목표는 어떤 위치의 주택들의 중간 가격(k$ 단위)이다.


### 사용법:

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

- __인자:__
    - __path__: 데이터셋을 로컬에 캐싱 해 둘 경로.
        (~/.keras/datasets 기준)
    - __seed__: 테스트 부분 계산 전의 데이터 뒤섞기를
        위한 난수 시드.
    - __test_split__: 테스트 셋으로 예약해 둘 데이터 부분의 비율.

- __반환:__
    Numpy 배열들의 튜플: `(x_train, y_train), (x_test, y_test)`.
