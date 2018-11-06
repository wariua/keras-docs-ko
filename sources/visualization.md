
## 모델 시각화

케라스 모델을 (`graphviz`로) 그려 주는 유틸리티 함수들을
`keras.utils.vis_utils` 모듈에서 제공한다.

다음처럼 하면 모델의 그래프를 그려서 파일로 저장한다.
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model`은 선택적인 인자 두 개를 받는다.

- `show_shapes`(기본값 False)는 그래프에 출력 형태를 표시할지를 제어한다.
- `show_layer_names`(기본값 True)는 그래프에 층 이름을 표시할지를 제어한다.

`pydot.Graph` 객체를 얻은 다음 그리기를 직접 할 수도 있다.
예를 들어 ipython 노트북에 표시하려면:
```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

## 훈련 이력 시각화

케라스 `Model`의 `fit()` 메소드는 `History` 객체를 반환한다. `History.history` 속성은 
연속된 에포크에서의 훈련 손실 값과 측정치 값, 그리고 (적용 가능한 경우) 검증 손실 값과
검증 측정치 값을 기록한 딕셔너리이다. 다음은 `matplotlib`를 이용해 훈련 및 검증에 대한 손실 및 정확도 그래프를 만들어 내는 간단한 예이다.

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 훈련 및 검증 정확도 값 그리기
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 훈련 및 검증 손실 값 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
