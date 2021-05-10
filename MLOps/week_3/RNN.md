# 04-12

# Lecture 3: Recurrent Neural Networks (Full Stack Deep Learning - Spring 2021)

- [https://www.youtube.com/watch?v=2b0TPDmzoaQ&t=784s](https://www.youtube.com/watch?v=2b0TPDmzoaQ&t=784s)

## Sequence Problems

- sequence Problems
    - time series
    - text
    - translation
    - speech recognition and generation
    - image captioning
    - question answering
- types of sequence problems
    - one to many
    - many to one
    - many to many

## RNNs

![04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled.png](04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled.png)

```python
class RNN:
		def compute_next_h(self, x):
				h = np.tanh(self.W_hh.dot(self.h) + self.W_xh.dot(x))
				return h
		def step(self, x):
				self.h = self.compute_next_h(x)
				y = self.W_hy.dot(self.h)
				return y
```

## Vanishing gradients and LSTMs

### Vanishing gradients

- vanilla RNNs
    - Can't handle more than 10~20 timesteps
    - longer-term dependencies get lost

### LSTMs

```python
class LSTM(RNN):
		def compute_next_h(self, x):
				h = lstm(x, self.h)
				return h
```

![04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%201.png](04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%201.png)

## Case study: Machine Translation(Bidirectionality and Attention)

### solution: attention

- Key idea: instead of compressing all past time steps into a single hidden state, give the neural network access to the entire history

### solution: bidirectionality

- Key idea: Use one LSTM to process the sequence in forward order, the other in backward order

## CTC loss

![04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%202.png](04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%202.png)

## Pros and Cons

### pros

- Encoder / decoder LSTM architectures can model arbitrary (one-to-many, many-to-one, and many-to-many) sequence problems
- Many successes in NLP and other applictations

### cons

- Recurrent network training is not as parallelizable as FC or CNN, due to the need to go in sequence

## A preview of non-recurrent sequence models

![04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%203.png](04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%203.png)

![04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%204.png](04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%204.png)

문제점: 너무 제한적인 input값으로 output을 만든다.

![04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%205.png](04-12%20f1a4da9996064909bdfa3107db5ea2e8/Untitled%205.png)

해결방안: input을 스킵하면서 본다.
