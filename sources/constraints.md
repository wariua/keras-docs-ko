## 제약 사용

`constraints` 모듈의 함수들을 통해 망 매개변수들에 대한 최적화 중 제약(가령, 음수 안 됨)을 설정할 수 있다.

층 단위로 패널티가 적용된다. 정확한 API는 층에 따라 다르겠지만 `Dense`, `Conv1D`, `Conv2D`, `Conv3D` 층에는 공통된 API가 있다.

그 층들에는 2가지 키워드 인자가 있다.

- 주 가중치 행렬에 대한 `kernel_constraint`
- 편향에 대한 `bias_constraint`


```python
from keras.constraints import max_norm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## 사용 가능한 제약

- __max_norm(max_value=2, axis=0)__: 최대 norm 제약
- __non_neg()__: 비음수 제약
- __unit_norm(axis=0)__: 단위 norm 제약
- __min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)__:  최소/최대 norm 제약
