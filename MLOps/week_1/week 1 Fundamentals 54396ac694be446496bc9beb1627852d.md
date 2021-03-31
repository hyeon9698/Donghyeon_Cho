# week 1: Fundamentals

The week of February 1, we do a blitz review of the fundamentals of deep learning, and introduce the codebase we will be working on in labs for the remainder of the class.

# **[Lecture 1: DL Fundamentals](https://fullstackdeeplearning.com/spring2021/lecture-1/)**

- Neural Networks
    - 머리 속에 있는 뉴련을 따라함

        ![week%201%20Fundamentals%2054396ac694be446496bc9beb1627852d/Untitled.png](week%201%20Fundamentals%2054396ac694be446496bc9beb1627852d/Untitled.png)

    - 1950년에 만들어짐 perceptron 이라 불리기도 함
    - W는 가중치(weight), b는 bias y축을 움직이는데 사용함
    - linear function임
    - activation function으로 감싸져 있음
    - 뇌의 뉴런처럼 자극의 합이 특정 값을 넘으면 output으로 쏘는 느낌임
    - activation function은 여러 종류가 있음
    - Sigmoid 함수 → input이 무엇이든 output을 0과 1 사이로 만듦
    - 최근에는 ReLU를 많이 사용함 Rectified Linear Unit
        - max function이라고도 함
        - input이 0을 넘으면 넘기고 아니면 넘기지 않는다

    ![week%201%20Fundamentals%2054396ac694be446496bc9beb1627852d/Untitled%201.png](week%201%20Fundamentals%2054396ac694be446496bc9beb1627852d/Untitled%201.png)

    - input layer → hidden layer 1~n → output layer
    - 각자 weight과 bias가 있다
- Universality, Universal Approximation Function
    - 2개의 neural networks 즉 input layer → hidden layer → output layer 구조에서는 hidden units가 충분히 주어지면 아무 function을 approximate 할 수 있다
    - fourier transform과 비슷하다
- Learning Problems
    - Supervised Learning
        - X와 Y 값을 동시에 가짐
        - 이미 답이 있는 것으로 학습을 하고 예측을 함
    - Unsupervised Learning
        - goal is to learn the structure of that data
        - you can generate more of theses types of data
        - fake sound clips images, reviews
        - obtain insights into what the data might hold
        - 잘 모르는 데이터를 가지고 함
        - GAN (Generative adversarial networks
    - Reinforcement Learning
        - environmnet에서 score을 최대화하는 action을 함
        - reward를 주면서 학습을 시킴
    - Transfer learning, imitation learning, meta-learning, ...
- Empirical Risk Minimization / Loss Functions
    - Linear Regression
        - input을 가지면 output은 어떤지 예측
        - 가장 만족하는 line을 만들려면 minimize the squared error을 할 수도 있다
        - 우리는 a loss fuinction → minimize 하고 싶어함
        - classification은 input을 가지고 categorical output을 예측함
            - cross entropy loss 을 많이 사용함
- Gradient Descent

    ![week%201%20Fundamentals%2054396ac694be446496bc9beb1627852d/Untitled%202.png](week%201%20Fundamentals%2054396ac694be446496bc9beb1627852d/Untitled%202.png)

    - Our goal: find w, b that optimize
    - batch gradient descent, stochastic gradient descent
        - compute each gradient step on just a subset "batch" of data
        - less compute per step
        - more noisy per step
        - faster progress per amount of compute
- Backpropagation / Automatic Differentiation
    - chain rule
    - automatic differentiation software을 사용함
        - pytorch, tensorflow, theano, chainer, etc
- Architectural Considerations (deep / conv / rnn)
    - computer vision → convolutional neural networks
    - natural language processing → recurrent networks
- CUDA / Cores of Compute
    - All neural networks computations are matrix multiplications, which are well parallelized onto the many cores of a GPU

# **[Notebook: Coding a neural net from scratch](https://fullstackdeeplearning.com/spring2021/notebook-1/)**

Google Colab 

[Redirecting](https://www.google.com/url?q=https://colab.research.google.com/drive/1HS3qbHArkqFlImT2KnF5pcMCz7ueHNvY?usp%3Dsharing&sa=D&ust=1611957782312000&usg=AOvVaw1Y_iMZT2dnCn82PilLI0H-)

# **[Lab 1: Setup and Intro](https://fullstackdeeplearning.com/spring2021/lab-1/)**

# **[How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)**