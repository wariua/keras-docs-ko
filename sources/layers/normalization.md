<span style="float:right;">[[소스]](https://github.com/keras-team/keras/blob/master/keras/layers/normalization.py#L16)</span>
### BatchNormalization

```python
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

배치 정규화 층 (Ioffe and Szegedy, 2014).

각 배치마다 앞 층의 활성을 정규화한다.
즉 평균 활성을 0 근처로 유지하고 활성 표준 편차를
1 근처로 유지하는 변환을 적용한다.

__인자__

- __axis__: 정수. 정규화 해야 할 축.
    (보통은 피쳐 축.)
    예를 들어 `data_format="channels_first"`인
    `Conv2D` 층 다음에 있는 `BatchNormalization`에선
    `axis=1`로 설정하면 된다.
- __momentum__: 이동 평균 및 이동 분산에 대한 모멘텀.
- __epsilon__: 0으로 나누기를 피하기 위해 분산에 더하는 작은 실수.
- __center__: True이면 `beta`의 오프셋을 정규화된 텐서에 더한다.
    False이면 `beta`를 무시한다.
- __scale__: True이면 `gamma`를 곱한다.
    False이면 `gamma`를 안 쓴다.
    다음 층이 선형일 때는 (그리고 가령 `nn.relu`일 때)
    그 층에서 스케일링이 이뤄지게 되므로
    이 옵션을 끌 수 있다.
- __beta_initializer__: beta 가중치의 initializer.
- __gamma_initializer__: gamma 가중치의 initializer.
- __moving_mean_initializer__: 이동 평균의 initializer.
- __moving_variance_initializer__: 이동 분산의 initializer.
- __beta_regularizer__: 선택적. beta 가중치의 정칙화.
- __gamma_regularizer__: 선택적. gamma 가중치의 정칙화.
- __beta_constraint__: 선택적. beta 가중치의 제약.
- __gamma_constraint__: 선택적. gamma 가중치의 제약.

__입력 형태__

마음대로. 이 층을 모델의 첫 번째 층으로 쓸 때는
키워드 인자 `input_shape`(정수들의 튜플, 표본 축은
제외)를 써야 한다.

__출력 형태__

입력과 같은 형태.

__참고 자료__

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
