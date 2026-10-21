import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(1 / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / hidden_size)
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        x_t = np.asarray(x_t, dtype=np.float64)
        h_prev = np.asarray(h_prev, dtype=np.float64)
        if x_t.shape != (self.input_size,):
            raise ValueError(f"x_t must have shape ({self.input_size},)")
        if h_prev.shape != (self.hidden_size,):
            raise ValueError(f"h_prev must have shape ({self.hidden_size},)")
        return np.tanh(self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h)
