import numpy as np
import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

    @property
    def seconds(self):
        return self.elapsed

    @property
    def minutes(self):
        return self.elapsed / 60


def compare_scalers(scaler1, scaler2):
    return np.allclose(scaler1.mean_, scaler2.mean_) and \
           np.allclose(scaler1.var_, scaler2.var_) and \
           np.allclose(scaler1.scale_, scaler2.scale_)
