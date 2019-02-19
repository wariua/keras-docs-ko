# 왜 케라스인가?

요즘은 쓸 수 있는 심층학습 프레임워크가 수도 없이 많다. 그렇다면 왜 굳이 케라스를 쓰는 걸까? 기존 대안들과 비교할 때 케라스가 더 알맞은 부분이 몇 가지 있다.

---

## 케라스에선 개발자 경험이 우선이다
 
- 케라스는 기계가 아니라 인간을 위해 설계된 API다. [케라스에서는 인지 부하를 줄여 주는 우수 관행들을 따른다](https://blog.keras.io/user-experience-design-for-apis.html). 즉 일관적이고 단순한 API를 제공하고, 빈번한 사용 사례들에 필요한 사용자 동작을 최소화 하고, 사용자 오류에 대해 분명하고 대처 가능한 피드백을 제공한다.
- 그래서 케라스는 배우기 쉽고 사용하기 쉽다. 케라스 사용자는 더 생산적이고, 그래서 경쟁자들보다 더 많은 아이디어를 더 빠르게 시도해 볼 수 있다. [머신 러닝 대회 우승에도 도움이 된다](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions).
- 이렇게 쓰기 쉽다고 해서 유연성이 떨어지지 않는다. 케라스는 저수준 심층학습 언어와 (특히 텐서플로우와) 밀접하게 연계돼 있기 때문에 그 기반 언어로 만들 수 있는 건 뭐든지 구현할 수 있다. 특히 `tf.keras`에서 보듯 케라스 API는 텐서플로우 작업 흐름에 매끄럽게 연결된다.

---

## 업계와 연구자 집단에서 케라스를 널리 채택하고 있다

<a href='https://towardsdatascience.com/deep-learning-framework-power-scores-2018-23607ddf297a'>
    <img style='width: 80%; margin-left: 10%;' src='https://s3.amazonaws.com/keras.io/img/dl_frameworks_power_scores.png'/>
</a>
<p style='font-style: italic; font-size: 10pt; text-align: center;'>
    심층학습 프레임워크 순위. Jeff Hale이 출처 11곳의 자료를 7가지 범주로 계산.
</i>

2018년 중반 기준 250,000명의 개인 사용자가 있는 케라스는 업계와 연구자 집단 모두에서 텐서플로우 자체를 제외하고 다른 어떤 심층학습 프레임워크보다 많이 채택되고 있다. (그리고 케라스 API는 `tf.keras` 모듈을 통해 사용할 수 있는 텐서플로우 공식 프론트엔드이다.)

사람들은 이미 케라스로 만들어진 기능들과 꾸준히 만나고 있다. Netflix, Uber, Yelp, Instacart, Zocdoc, Square, 기타 많은 업체에서 케라스를 쓴다. 특히 제품 핵심에 심층학습이 있는 스타트업들에게 인기가 많다.

심층학습 연구자들도 케라스를 좋아해서 [arXiv.org](https://arxiv.org/archive/cs) 서버에 올라오는 과학 논문들에서 두 번째로 많이 언급된다. 또 CERN이나 NASA 같은 대규모 과학 조직에서도 연구자들이 케라스를 채택하고 있다.

---

## 케라스에서는 모델을 제품으로 바꾸는 게 쉽다

케라스 모델은 다른 어떤 심층학습 프레임워크보다 다양한 플랫폼들로 쉽게 도입할 수 있다.

- iOS에는 [Apple의 CoreML](https://developer.apple.com/documentation/coreml)이 있다. (Apple에서 케라스를 공식적으로 지원한다.) [튜토리얼](https://www.pyimagesearch.com/2018/04/23/running-keras-models-on-ios-with-coreml/)이 있다.
- 안드로이드에는 텐서플로우 안드로이드 런타임이 있다. 예시: [Not Hotdog 앱](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3).
- 브라우저에는 [Keras.js](https://transcranial.github.io/keras-js/#/)나 [WebDNN](https://mil-tokyo.github.io/webdnn/) 같은 GPU 가속 자바스크립트 런타임이 있다.
- 구글 클라우드에는 [TensorFlow-Serving](https://www.tensorflow.org/serving/)이 있다.
- [(플라스크 앱 같은) 파이썬 웹앱 백엔드에도](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html).
- JVM에는 [SkyMind에서 제공하는 DL4J 모델 가져오기](https://deeplearning4j.org/model-import-keras)가 있다.
- 라즈베리 파이에도.

---

## 케라스에서는 여러 백엔드 엔진을 지원하므로 사용자를 한 생태계에 가두지 않는다

다양한 [심층학습 백엔드들](https://keras.io/backend/)을 이용해 케라스 모델을 개발할 수 있다. 내장 층들만 활용하는 케라스 모델은 그 백엔드들 모두에 이식성이 있다. 즉 모델을 어떤 백엔드로 훈련시킨 다음 (가령 도입 시에) 다른 모델에서 적재할 수 있다. 다음 백엔드를 사용할 수 있다.

- 텐서플로우 백엔드 (구글)
- CNTK 백엔드 (마이크로소프트)
- 테아노 백엔드

아마존에서도 케라스를 위한 MXNet 백엔드 개발 작업을 하고 있다.

또한 CPU뿐 아니라 다양한 하드웨어에서 케라스 모델을 훈련시킬 수 있다.

- [NVIDIA GPU](https://developer.nvidia.com/deep-learning)
- [구글 TPU](https://cloud.google.com/tpu/), 텐서플로우 백엔드 및 구글 클라우드 사용
- AMD 등의 OpenCL 사용 GPU, [PlaidML 케라스 백엔드](https://github.com/plaidml/plaidml) 사용

---

## 케라스에서는 다중 GPU 및 분산 훈련을 확실히 지원한다

- 케라스에는 [다중 GPU 데이터 병렬화 지원이 내장](/utils/#multi_gpu_model)돼 있다.
- Uber의 [Horovod](https://github.com/uber/horovod)에서 케라스 모델을 일등급으로 지원한다.
- 케라스 모델을 [텐서플로우 Estimator로 바꿔서](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/estimator/model_to_estimator) [구글 클라우드의 GPU 클러스터](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)에서 훈련시킬 수 있다.
- (CERN의) [Dist-Keras](https://github.com/cerndb/dist-keras)와 [Elephas](https://github.com/maxpumperla/elephas)를 통해 Spark에서 케라스를 돌릴 수 있다.

---

## 심층학습 생태계의 주요 업체들이 케라스 개발을 지원한다

주로 구글이 케라스 개발을 돕고 있으며 텐서플로우에 케라스 API가 `tf.keras`로 패키징 돼 있다. 더불어 마이크로소프트에서 CNTK 케라스 백엔드를 유지하고 있다. 또 아마존 AWS에서 MXNet 지원을 개발하고 있다. 도움을 주는 다른 회사들로 NVIDIA, Uber, Apple (CoreML) 등이 있다.

<img src='/img/google-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/microsoft-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/nvidia-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/aws-logo.png' style='width:110px; margin-right:15px;'/>
