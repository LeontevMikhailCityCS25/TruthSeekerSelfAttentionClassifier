import numpy as np

class SingleHeadSelfAttention:
    def __init__(self, input_dim):
        self.W_q = np.random.randn(input_dim, input_dim)
        self.W_k = np.random.randn(input_dim, input_dim)
        self.W_v = np.random.randn(input_dim, input_dim)

    def forward(self, X):
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        scores = np.matmul(Q, K.T) / np.sqrt(X.shape[1])
        weights = self.softmax(scores)
        output = np.matmul(weights, V)
        return output

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        normalized = e_x / e_x.sum(axis=-1, keepdims=True)
        return normalized

    # python
    def backward(self, d_out, X, learning_rate=0.001):
        # Forward pass (recompute for gradients)
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        scores = np.matmul(Q, K.T) / np.sqrt(X.shape[1])
        weights = self.softmax(scores)

        # Gradient of Values (V matrix)
        dV = np.matmul(weights.T, d_out)
        d_weights = np.matmul(d_out, V.T)

        # Gradient of softmax (scores)
        dscores = d_weights * weights * (1 - weights)

        # Gradients of Query and Key matrices (Q and K)
        dQ = np.matmul(dscores, K) / np.sqrt(X.shape[1])
        dK = np.matmul(dscores.T, Q) / np.sqrt(X.shape[1])

        # final Gradients weights
        dW_q = np.matmul(X.T, dQ)
        dW_k = np.matmul(X.T, dK)
        dW_v = np.matmul(X.T, dV)

        # Updated weights
        self.W_q -= learning_rate * dW_q
        self.W_k -= learning_rate * dW_k
        self.W_v -= learning_rate * dW_v


class MultiHeadSelfAttention:
    def __init__(self, input_dim, num_heads):
        assert input_dim % num_heads == 0, "Incorrect number of dimensions for the number of heads"
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.heads = [SingleHeadSelfAttention(self.head_dim) for _ in range(num_heads)]
        self.W_o = np.random.randn(input_dim, input_dim)

    def forward(self, X):
        X_split = np.split(X, self.num_heads, axis=-1)
        head_outputs = [head.forward(x) for head, x in zip(self.heads, X_split)]
        concatenated = np.concatenate(head_outputs, axis=-1)
        output = np.matmul(concatenated, self.W_o)
        return output