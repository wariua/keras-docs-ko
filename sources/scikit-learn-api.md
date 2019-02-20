# Scikit-Learn API 래퍼

`keras.wrappers.scikit_learn.py`에 있는 래퍼들을 통해 케라스의 `Sequential` 모델(단일 입력)을 Scikit-Learn 작업물에 포함시켜 쓸 수 있다.

두 가지 래퍼가 있다.

`keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)`: Scikit-Learn 분류 인터페이스 구현

`keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)`: Scikit-Learn 회귀 인터페이스 구현

### 인자

- __build_fn__: 호출 가능 함수 또는 클래스 인스턴스
- __sk_params__: 모델 매개변수 및 훈련 매개변수

`build_fn`에서 케라스 모델을 구성하고 컴파일 해서 반환해야 한다.
그러면 그걸 훈련/예측에 쓰게 된다. `build_fn`으로 다음 세 가지
값 중 하나를 줄 수 있다.

1. 함수
2. `__call__` 메소드를 구현한 클래스의 인스턴스
3. None. `KerasClassifier`나 `KerasRegressor`를 상속하는
클래스를 구현한다는 뜻이다. 그 기존 클래스의 `__call__`
메소드를 기본 `build_fn`으로 처리하게 된다.

`sk_params`는 모델 매개변수와 훈련 매개변수를 모두 받는다. 모델
매개변수로 쓸 수 있는 건 `build_fn`의 인자들이다. 참고로 Scikit-Learn의
다른 모든 estimator들처럼 `build_fn`에서도 인자들에 기본값을
제공해서 `sk_params`에 아무 값도 주지 않고도 estimator를 만들
수 있도록 해야 한다.

`sk_params`는 `fit`, `predict`, `predict_proba`, `score` 메소드의
매개변수들(가령 `epochs`, `batch_size`)을 받을 수도 있다.
훈련 (예측) 매개변수들을 다음 순서로 선택한다.

1. `fit`, `predict`, `predict_proba`, `score` 메소드의
딕셔너리 인자로 전달된 값.
2. `sk_params`로 전달된 값.
3. `keras.models.Sequential`의 `fit`, `predict`,
`predict_proba`, `score` 메소드의 기본값.

Scikit-Learn의 `grid_search` API 사용 시 튜닝 가능한 매개변수들은
훈련 매개변수들을 포함해 `sk_params`로 전달할 수 있는 매개변수들이다.
달리 말해 `grid_search`를 이용해 모델 매개변수뿐 아니라 최적의
`batch_size`나 `epochs`도 탐색할 수 있다.
