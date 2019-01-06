
### 텍스트 전처리

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py#L137)</span>
### Tokenizer

```python
keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ', char_level=False, oov_token=None)
```

텍스트를 토큰으로 나누는 유틸리티 클래스.

이 클래스를 이용해 텍스트 말뭉치를 벡터로 만들 수 있다.
각 텍스트를 정수 열(각 정수가 딕셔너리 내 토큰의 인덱스)로,
또는 각 토큰에 대한 계수가 바이너리이거나, 단어 수 기준이거나,
tf-idf 기준인 벡터로 바꾼다.

__인자__

- __num_words__: 단어 빈도 기준 유지 단어 최대 개수.
    가장 자주 나오는 `num_words` 개 단어만 유지한다.
- __filters__: 텍스트에서 걸러내야 할 문자들로 된 문자열.
    기본값은 문장 부호 전체에 탭과 개행을 더하고,
    `'` 문자는 뺀 것이다.
- __lower__: 불리언. 텍스트를 소문자로 바꿀지 여부.
- __split__: 문자열. 단어 나누기 구분자.
- __char_level__: True이면 모든 문자를 토큰으로 취급한다.
- __oov_token__: word_index에 추가해서 text_to_sequence 호출에서
    어휘 외(out-of-vocabulary) 단어들을 이 토큰으로 대체한다.

기본적으로 모든 문장 부호를 제거해서 텍스트를 공백으로 구분된
단어 열로 바꾼다. (단어에 `'` 문자가 포함돼 있을 수 있다.)
그리고 그 열을 토큰들의 리스트로 쪼개고,
인덱스 내지 벡터로 바꾼다.

`0`은 예약된 인덱스이며 아무 단어에도 할당되지 않는다.

----

### hashing_trick


```python
keras.preprocessing.text.hashing_trick(text, n, hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```


텍스트를 고정 크기 해싱 공간 인덱스 열로 바꾸기.

__인자__

- __text__: 입력 테스트 (문자열).
- __n__: 해싱 공간의 차원.
- __hash_function__: 기본은 파이썬 `hash` 함수이며, 'md5' 또는
    문자열을 받아서 정수를 반환하는 어떤 함수도 가능하다.
    'hash'는 안정적 해시 함수가 아니므로 실행 결과가
    일관적이지 않다. 반면 'md5'는 안정적인 해시 함수다.
- __filters__: 문장 부호들처럼 걸러내야 할 문자들을 이어 붙인 목록.
    기본값: 기본 부호 ``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~``, 탭, 개행 포함.
- __lower__: 불리언. 텍스트를 소문자로 바꿀지 여부.
- __split__: 문자열. 단어 나누기 구분자.

__반환__

정수 단어 인덱스들의 리스트. (일대일 관계 보장 안 됨.)

`0`은 예약된 인덱스이며 아무 단어에도 할당되지 않는다.

해싱 함수에서 발생 가능한 충돌 때문에 둘 이상의 단어가
같은 인덱스에 할당될 수 있다.
충돌 [확률](https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)은
해싱 공간 차원 및 구별되는 객체 수와 관계 있다.

----

### one_hot


```python
keras.preprocessing.text.one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```


텍스트를 크기 n인 단어 인덱스 리스트로 원핫 인코딩.

해싱 함수로 `hash`를 사용해 `hashing_trick` 함수를 감싼 것이다.
단어에서 인덱스로의 매핑에 일대일 관계가 보장되지 않는다.

__인자__

- __text__: 입력 테스트 (문자열).
- __n__: 정수. 어휘 크기.
- __filters__: 문장 부호들처럼 걸러내야 할 문자들을 이어 붙인 목록.
    기본값: 기본 부호 ``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~``, 탭, 개행 포함.
- __lower__: 불리언. 텍스트를 소문자로 바꿀지 여부.
- __split__: 문자열. 단어 나누기 구분자.

__반환__

[1, n] 범위 정수들의 리스트. 각 정수가 단어를 나타냄.
(일대일 관계 보장 안 됨.)

----

### text_to_word_sequence


```python
keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```


텍스트를 단어(토큰) 열로 바꾸기.

__인자__

- __text__: 입력 테스트 (문자열).
- __filters__: 문장 부호들처럼 걸러내야 할 문자들을 이어 붙인 목록.
    기본값: 기본 부호 ``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~``, 탭, 개행 포함.
- __lower__: 불리언. 입력을 소문자로 바꿀지 여부.
- __split__: 문자열. 단어 나누기 구분자.

__반환__

단어(토큰)들의 리스트.

