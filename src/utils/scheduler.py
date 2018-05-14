class Scheduler(object):
    def __init__(self, v, n_values):
        """
        線形に推移する値を管理するクラス．
        指定の時間をかけて0まで線形に推移．
        :param v:           初期値
        :param n_values:    0に推移するまでのステップ数
        """
        self.n = 0.
        self.v = v
        self.n_values = n_values

    def value(self):
        """
        ステップ数を更新し，それに応じた値を取得
        """
        current_value = self.v * (1.0 - self.n / self.n_values)
        self.n += 1.
        return current_value
