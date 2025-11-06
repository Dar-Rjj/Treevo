import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price-volume divergence with spectral decomposition
    price_velocity = close.diff(3) / close.shift(3)
    volume_velocity = volume.diff(3) / volume.shift(3)
    pv_divergence = (price_velocity - volume_velocity.rolling(8).mean()) * np.sign(price_velocity)
    
    # Regime-switching momentum with entropy-based persistence
    short_entropy = -close.pct_change(3).abs().rolling(5).apply(lambda x: np.sum(x * np.log(x + 1e-8)))
    long_entropy = -close.pct_change(10).abs().rolling(10).apply(lambda x: np.sum(x * np.log(x + 1e-8)))
    regime_persistence = (short_entropy - long_entropy) / (short_entropy + long_entropy + 1e-8)
    
    # Liquidity-adjusted volatility with fractal dimension
    hl_range = (high - low) / close
    dollar_volume = volume * close
    liquidity_vol = hl_range.rolling(8).std() * (1 - dollar_volume.rank(pct=True))
    
    # Spectral momentum through frequency domain analysis
    price_series = close.rolling(13).apply(lambda x: np.fft.fft(x - x.mean()).real[1:4].mean())
    volume_series = volume.rolling(13).apply(lambda x: np.fft.fft(x - x.mean()).real[1:4].mean())
    spectral_momentum = price_series * volume_series
    
    # Entropy-weighted factor combination
    factors = pd.DataFrame({
        'divergence': pv_divergence,
        'regime': regime_persistence,
        'liquidity': liquidity_vol,
        'spectral': spectral_momentum
    })
    
    entropy_weights = []
    for col in factors.columns:
        series = factors[col].dropna()
        if len(series) > 0:
            hist = np.histogram(series, bins=min(20, len(series)//10))[0]
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log(prob + 1e-8))
            entropy_weights.append(entropy)
        else:
            entropy_weights.append(1.0)
    
    weights = np.array(entropy_weights) / sum(entropy_weights)
    heuristics_matrix = pd.Series(0, index=df.index)
    
    for i, col in enumerate(factors.columns):
        heuristics_matrix += factors[col] * weights[i]
    
    return heuristics_matrix
