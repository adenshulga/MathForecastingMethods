import pandas as pd
import numpy as np

def np_corr(f: np.ndarray, g: np.ndarray) -> float:

    return f@g / (np.sqrt(f@f) * np.sqrt(g@g) )


class ProximityEstimator:
    
    def __init__(self, k: int) -> None:
        self.k = k  # sequence size which we use for distance evaluation

    def fit(self, ts: np.ndarray) -> None:
        self.ts = ts
        self.len = len(ts)

    def evaluate(self, index: int):
        assert (self.k < index < self.len)
        segment_for_prediction = self.ts[index - self.k:index]
        corr_arr = []
        for j in range(index - self.k + 1):
            segment = self.ts[j:j + self.k]
            corr = np.corrcoef(segment_for_prediction, segment)[0, 1]  # Get the correlation value
            corr_arr.append(corr)

        segment_begin = np.argmax(corr_arr)

        return self.ts[segment_begin + self.k]

    def predict(self):
        prediction = [self.ts[i] for i in range(self.k + 1)]
        for i in range(self.k + 1, self.len):
            next_val = self.evaluate(i)
            prediction.append(next_val)

        return np.array(prediction)

class ConstantDependenciesRetrieval:
    
    def __init__(self, mask: np.ndarray, k) -> None:
        
        self.k = k
        self.mask = mask
        self.N = len(mask)
        self.mask_power = np.sum(mask)

        self.freq_tensor = np.zeros([128 for i in range(self.mask_power + 1)])

    def fit(self, ts: np.ndarray) -> None:
        self.ts = ts


    def get_next_value(self, signature: tuple) -> int:
        
        vm = self.freq_tensor[signature]
        vmax = np.max(vm)
        v2max = np.partition(vm, -2)[-2]
        valpha = np.sum(vm)

        if vmax - v2max > self.k * valpha:
            return np.argmax(vm)
        else:
            return np.nan
        

    def evaluate(self, i: int) -> None:
        args = self.ts[i - self.N: i]
        # костыль, нулевая нота не используется, но все равно плохо конечно
        values = list(args[np.nonzero(args * self.mask)[0]])
        values.append(self.ts[i])
        self.freq_tensor[tuple(values)] += 1


    def predict(self):
        
        predictions = [self.ts[i] for i in range(self.N + 1)]

        for i in range(self.N, len(self.ts) - 1):
            self.evaluate(i)
            args = self.ts[i-self.N+1:i+1]
            sign = tuple(args[np.nonzero(args * self.mask)[0]])
            predictions.append(self.get_next_value(sign))

        return np.array(predictions)















