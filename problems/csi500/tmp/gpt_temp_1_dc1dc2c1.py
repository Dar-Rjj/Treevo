import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple heuristics:
    - Correlation Decay Momentum
    - Volume Regime Breakout  
    - Amplitude-Modulated Volatility
    - Liquidity-Adjusted Momentum Divergence
    - Volatility Clustering Persistence
    - Order Flow Imbalance Momentum
    - Price-Volume Fractal Dimension
    - Regime-Switching Mean Reversion
    """
    
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Correlation Decay Momentum
    corr_short = data['close'].rolling(5).corr(data['volume'])
    corr_long = data['close'].rolling(20).corr(data['volume'])
    corr_decay = (corr_short - corr_long).ewm(span=10).mean()
    momentum_5d = data['close'].pct_change(5)
    corr_momentum = corr_decay * momentum_5d
    
    # 2. Volume Regime Breakout
    vol_percentile = data['volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    vol_regime = (vol_percentile > 0.8).astype(int) - (vol_percentile < 0.2).astype(int)
    high_5d = data['high'].rolling(5).max()
    low_5d = data['low'].rolling(5).min()
    breakout_up = (data['close'] > high_5d.shift(1)).astype(int)
    breakout_down = (data['close'] < low_5d.shift(1)).astype(int)
    breakout_signal = (breakout_up - breakout_down) * vol_regime
    
    # 3. Amplitude-Modulated Volatility
    daily_range = (data['high'] - data['low']) / data['close']
    vol_anomaly = (data['volume'] / data['volume'].rolling(20).mean() - 1)
    amplitude_signal = daily_range * vol_anomaly
    amplitude_signal = amplitude_signal.rolling(5).mean()
    
    # 4. Liquidity-Adjusted Momentum Divergence
    momentum_10d = data['close'].pct_change(10)
    momentum_20d = data['close'].pct_change(20)
    liquidity_score = (data['amount'] / data['volume']).rolling(10).mean()
    liquidity_norm = (liquidity_score - liquidity_score.rolling(20).mean()) / liquidity_score.rolling(20).std()
    momentum_divergence = (momentum_10d - momentum_20d) * liquidity_norm
    
    # 5. Volatility Clustering Persistence
    volatility = data['close'].pct_change().rolling(5).std()
    vol_clustering = volatility.rolling(10).apply(lambda x: x.autocorr())
    vol_persistence = volatility.rolling(10).apply(lambda x: (x > x.mean()).sum() / len(x))
    clustering_signal = vol_clustering * vol_persistence * data['close'].pct_change(3)
    
    # 6. Order Flow Imbalance Momentum
    price_trend = np.sign(data['close'].diff(3))
    amount_per_share = data['amount'] / data['volume']
    ofi = (amount_per_share - amount_per_share.rolling(10).mean()) / amount_per_share.rolling(10).std()
    ofi_momentum = ofi * price_trend
    
    # 7. Price-Volume Fractal Dimension (simplified)
    price_vol_corr = data['close'].rolling(10).corr(data['volume'])
    efficiency_ratio = abs(data['close'].diff(10)) / data['close'].diff().abs().rolling(10).sum()
    fractal_signal = price_vol_corr * efficiency_ratio
    
    # 8. Regime-Switching Mean Reversion
    trend_strength = data['close'].rolling(10).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() > 0 else 0)
    mean_reversion = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
    regime_weight = 1 / (1 + abs(trend_strength))
    regime_signal = mean_reversion * regime_weight
    
    # Combine all signals with equal weights
    signals = pd.DataFrame({
        'corr_momentum': corr_momentum,
        'breakout': breakout_signal,
        'amplitude': amplitude_signal,
        'momentum_div': momentum_divergence,
        'clustering': clustering_signal,
        'ofi_momentum': ofi_momentum,
        'fractal': fractal_signal,
        'regime': regime_signal
    })
    
    # Normalize each signal
    signals_normalized = signals.apply(lambda x: (x - x.rolling(50).mean()) / x.rolling(50).std())
    
    # Equal-weighted combination
    combined_signal = signals_normalized.mean(axis=1)
    
    return combined_signal
