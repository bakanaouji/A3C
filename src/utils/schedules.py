class LinearSchedule(object):
    def __init__(self, v, n_values):
        self.n = 0.
        self.v = v
        self.n_values = n_values

    def value(self):
        current_value = self.v * (1.0 - self.n / self.n_values)
        self.n += 1.
        return current_value
