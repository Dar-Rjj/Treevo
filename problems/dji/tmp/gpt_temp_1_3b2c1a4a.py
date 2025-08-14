importance of price movement over trading volume.}

```python
import numpy as np

def heuristics_v2(df):
    heuristics_matrix = (0.7 * df['close'].pct_change().shift(-1) + 0.3 * np.log(df['volume'])).dropna()
    return heuristics_matrix
