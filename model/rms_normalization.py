import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        s = sum((((j) ** 2) + eps for j in x)) / len(x)
        rms = math.sqrt(s)
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = round(gamma[i]*(x[i]/rms),4)
        return res
