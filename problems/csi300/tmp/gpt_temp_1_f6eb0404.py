import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-Regime Adaptive Momentum Factor
    # Matches momentum windows to volatility regimes with volume divergence decay
    # Interpretable as: Stocks with volatility-appropriate momentum signals confirmed by volume divergence
    
    # Calculate daily returns and volatility regimes
    returns = df['close'].pct_change()
    volatility_short = returns.rolling(window=5).std()
    volatility_long = returns.rolling(window=20).std()
    
    # Define volatility regimes (high vs low volatility)
    volatility_regime = volatility_short / volatility_long
    
    # Adaptive momentum windows based on volatility regime
    # Shorter windows in high volatility, longer in low volatility
    momentum_window = np.where(volatility_regime > 1, 3, 10)
    
    # Calculate adaptive momentum
    adaptive_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= max(momentum_window):
            window = int(momentum_window[i])
            adaptive_momentum.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-window]) / df['close'].iloc[i-window]
    
    # Volume divergence calculation
    volume_ma_short = df['volume'].rolling(window=5).mean()
    volume_ma_long = df['volume'].rolling(window=20).mean()
    volume_divergence = (volume_ma_short - volume_ma_long) / volume_ma_long
    
    # Weight decay by volume divergence
    # Positive divergence strengthens signal, negative weakens it
    volume_weight = 1 + volume_divergence
    
    # Volatility-normalized momentum
    volatility_normalized_momentum = adaptive_momentum / (volatility_short + 1e-7)
    
    # Combine volatility-normalized momentum and volume multiplicatively
    alpha = volatility_normalized_momentum * volume_weight
    
    return alpha
