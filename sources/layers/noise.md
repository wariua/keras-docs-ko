<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L14)</span>
### GaussianNoise

```python
keras.layers.GaussianNoise(stddev)
```

0이 중심인 가법 가우스 잡음 적용.

과적합 완화에 쓸모가 있다.
(일종의 무작위 데이터 증강으로 볼 수도 있다.)
가우스 잡음(Gaussian Noise, GS)은
현실 값의 입력에 대한 변형 과정으로
잘 맞는 선택지이다.

정칙화 층이므로 훈련 시에만 활성화된다.

__인자__

- __stddev__: float. 잡음 분포의 표준 편차.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 사용하라.

__출력 형태__

입력과 같은 형태.

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L58)</span>
### GaussianDropout

```python
keras.layers.GaussianDropout(rate)
```

1이 중심인 승법 가우스 잡음 적용.

정칙화 층이므로 훈련 시에만 활성화된다.

__인자__

- __rate__: float. (`Dropout`에서와 같은) 버리기 확률.
    승법 잡음의 표준 편차가
    `sqrt(rate / (1 - rate))`가 된다.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 사용하라.

__출력 형태__

입력과 같은 형태.

__참고 자료__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L105)</span>
### AlphaDropout

```python
keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
```

입력에 알파 드롭아웃 적용.

알파 드롭아웃은 입력의 평균과 분산을 원래 값으로 유지해서
이 드롭아웃을 거친 후에도 자체 정규화 특성이 보장되도록 하는
`Dropout`이다.
알파 드롭아웃은 활성을 무작위로 음수 포화 값으로 설정하므로
조정 지수 선형 단위에 잘 맞는다.

__인자__

- __rate__: float. (`Dropout`에서와 같은) 버리기 확률.
    승법 잡음의 표준 편차가
    `sqrt(rate / (1 - rate))`가 된다.
- __seed__: 난수 시드로 사용할 파이썬 정수.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 사용하라.

__출력 형태__

입력과 같은 형태.

__참고 자료__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
