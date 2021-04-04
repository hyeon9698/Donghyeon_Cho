# ResNet: Deep Residual Learning for Image Recognition

Blog: https://blog.naver.com/laonple/220761052425
Date: Apr 2, 2021
Explained Link: https://www.youtube.com/watch?v=671BsKl8d0E&t=883s
Paper: https://arxiv.org/abs/1512.03385
Tags: SAI-CV, vision, youtube, 동빈나
참고 링크: https://www.youtube.com/watch?v=671BsKl8d0E&t=883shttps://lv99.tistory.com/25,

잔여 학습(residual learning) 개념을 이용하여 모델의 최적화 난이도를 낮추어 아주 깊은 네트워크를 이용해도 학습이 잘 이루어질 수 있도록 합니다.

![ResNet%20Deep%20Residual%20Learning%20for%20Image%20Recognitio%20036f3bbf5b60430fa14e5b4fcdc6c0ff/Untitled.png](ResNet%20Deep%20Residual%20Learning%20for%20Image%20Recognitio%20036f3bbf5b60430fa14e5b4fcdc6c0ff/Untitled.png)

- 성능이 매우 뛰어남
- 논문에서 제안한 아이디어가 매우 쉬움
- 네트워크를 깊게 만들기 위해서 잔여 학습(residual learning)을 제안합니다.
- 기본적으로 네트워크가 깊어지면 풍부한 데이터를 추출할 수 있고 높은 성능을 기대할 수 있지만 오히려 layer이 너무 깊어지면 오히려 성능이 떨어지는 것을 볼 수 있다.

layer의 깊이가 너무 깊어지면 성능이 왜 떨어지는가?

그 문제를 해결한게 Resnet이라면 정말 layer의 깊이를 끝없이 깊게 하면 성능이 더 좋아지나?

- layer의 깊이가 길어지면 대표적으로 vanishing/exploding gradients 문제가 있다.
    - vanishing → sigmoid 활성 함수를 사용하면 layer이 깊으면 역전파 과정에서 기울기가 0에 가까워져서 학습이 잘 안됨
    - overfitting도 layer이 깊어지면 발생하는 문제이다.
- 위 그림을 보면 layer의 깊이가 높을때 오히려 성능이 떨어지는 것을 확인할 수 있다.
- 단순히 layer만 깊게 만들려면 identity mapping만 증가시키면 되지않을까?

![ResNet%20Deep%20Residual%20Learning%20for%20Image%20Recognitio%20036f3bbf5b60430fa14e5b4fcdc6c0ff/Untitled%201.png](ResNet%20Deep%20Residual%20Learning%20for%20Image%20Recognitio%20036f3bbf5b60430fa14e5b4fcdc6c0ff/Untitled%201.png)

- 본 논문에서는 단순히 layer을 깊게 쌓으면 어느 시점에서부터는 성능이 더 떨어지는 것을 잔여학습(residual learning)을 이용해서 해결한다.
- 본 논문의 내용은 기본적으로 CNN의 개념을 알고 있다는 전제로 전개가 된다.
- VGG 네트워크는 작은 크기의 3x3 컨볼루션 필터(filter)를 이용해 레이어의 깊이를 늘려 우수한 성능을 보인다.
    - 그러면 깊이를 계속 늘리면 성능이 좋지 않을까? → 아니였다

![ResNet%20Deep%20Residual%20Learning%20for%20Image%20Recognitio%20036f3bbf5b60430fa14e5b4fcdc6c0ff/Untitled%202.png](ResNet%20Deep%20Residual%20Learning%20for%20Image%20Recognitio%20036f3bbf5b60430fa14e5b4fcdc6c0ff/Untitled%202.png)

- 잔여 블록(residual block)을 이용해서 네트워크의 최적화(optimization) 난이도를 낮춥니다.
- 실제로 내재한 mapping인 H(x)를 곧바로 학습하는 것은 어려우므로 대신 F(x) = H(x) - x 를 학습합니다.
- 일반적으로 x input이 들어왔을때 weight layer은 convolutional neural network을 의미하고 convolutional neural network 연산을 통해서 하나의 특징들을 추출한 다음 relu와 같은 activation function을 거쳐서 전체 network가 nonlinear한 동작을 수행할 수 있도록 만들었음.

nonlinear한 동작을 수행할 수 있도록 만들었다는것이 잘 이해가 안됐음

- input 값 x 를 마지막 부분에 더해주기만 했는데 결과적으로 네트워크가 학습이 빠르고 정확하게 됐다. weight layer을 여러번 거쳐서 나온 결과값을 F(x)인거고 거기다 x를 더해준 형태가 우리가 의도했던 H(x)와 같은 값이 되도록 유도를 해준다. 앞의 학습된 정보 x를 그대로 가지고 올 수 있다. 거기에 추가적으로 F(x)를 더해준다 . 잔여한 정보인 F(x)만 학습할 수 있는 형태로 만들 수만 있으면 H(x)보다 F(x)가 학습시키기 더 쉬워서 오른쪽의 형태로 네트워크로 바꾸면 더욱 빠르고 더욱 높은 성능으로 잘 만들어진다.
- F(x)가 0이 되도록 하는게 학습 난이도가 쉽다. 극단적으로 H(x)가 x인 경우 residual 이 0이 되게하는게 쉽다.
- short-cut connections을 identity mapping으로 사용할 수 있다.
- 출력값에 x를 별도로 더해주는 것으로 끝내기 때문에 추가적인 parameter이나 computational complexity가 들지 않는다. 구현도 간단하다.

identity mapping 다시 공부하기

shortcut connections다시 공부하기

어쩌다가 입력값을 다시 더하는 생각을 하게 된걸까

![ResNet%20Deep%20Residual%20Learning%20for%20Image%20Recognitio%20036f3bbf5b60430fa14e5b4fcdc6c0ff/Untitled%203.png](ResNet%20Deep%20Residual%20Learning%20for%20Image%20Recognitio%20036f3bbf5b60430fa14e5b4fcdc6c0ff/Untitled%203.png)

- F(x) 함수를 오른쪽 그림처럼 정의했다. input x에 first weight layer 값을 곱하고 activation function relu를 씌워주고 그 다음 weight layer을 곱한 상태이다.
- ImageNet test에서 1등을 했다.
- x를 더함으로 기본적으로 identity mapping을 항상 수행할 수 있게 한다.
- 물론 실제 optimal한 솔루션이 identity mapping일 확률은 희박하다. 하지만 문제를 더 쉽게 해결해줄 수 있다.
- optimal한 function이 zero mapping보다는 identity mapping에 가깝다면 residual function을 이용할때 학습할때 편하다

zero mapping 다시 공부하기

y = F(x, {Wi}) + Wsx.

- x를 short-cut connection을 이용해서 mapping을 시켜줄때 input dimension과 output dimension이 일치하지 않는다면 x에 Ws를 곱해주면서 프로젝션을 시켜줌으로 matching을 시켜줄 수 있다.
- F를 만들때 weight값을 하나로 이용하게 되면 하나의 linear이기때문에 이득을 취하기는 어렵다.
- VGG와 비교했을때 FLOPs가 더 낮았다.
    - FLOPs 계산 복잡도의 척도

FLOPs 공부하기

- 입력과 출력의 dimension이 같다면 identity shortcuts를 사용할 수 있고, 그렇지 않다면 2가지 옵션이 있다.
    1. 사이드에 padding 붙히고 identity를 한다.
    2. projection을 이용한 shortcut을 사용해서 dimension을 바꾼다.
- 매 convolution 이후와 activation이전에 batch normalization을 한다.

batch normalization도 다시 공부하기

- learning rate도 0.1로 시작해서 점점 줄여나간다.
- weight decay와 momentum
- dropout은 사용하지 않는다.

weight decay와 momentum, dropout 공부하기

- signal 값들이 vanishing하는 문제는 거의 없었으며 수렴률이 기하급수적으로 낮아지는 것이 문제다.
- convergence rates → 최적화 기법 수렵을 위해 필요한 epoch이나 수렴 난이도를 언급할때 사용되는 척도

convergence rate 공부하기

- 초기 단계에서 더욱 빠르게 수렴하게 만듦
- bottleneck architectures에서는 복잡도를 증가시키지 않기 위해 더욱 효과적으로 사용할 수 있다.

bottleneck architectures가 무엇인가

- bottleneck design에서는 identity 방식이 더욱 효과적일 수 있다.
-