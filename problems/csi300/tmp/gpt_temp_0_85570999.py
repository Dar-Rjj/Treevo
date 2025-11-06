import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors for stock return prediction using multiple approaches
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    # 1. Momentum Decay Adjusted by Volume Skewness
    # Calculate 10-day and 20-day price momentum
    momentum_10 = data['close'].pct_change(periods=10)
    momentum_20 = data['close'].pct_change(periods=20)
    
    # Calculate volume skewness over 20-day window
    volume_skewness = data['volume'].rolling(window=20).skew()
    
    # Apply exponential decay weighted by volume skewness sign
    decay_factor = np.exp(-0.1 * np.arange(1, 21))
    weighted_momentum = (momentum_10 * 0.6 + momentum_20 * 0.4) * np.sign(volume_skewness)
    momentum_factor = weighted_momentum.rolling(window=20).apply(
        lambda x: np.sum(x * decay_factor[:len(x)]) if len(x) == 20 else np.nan
    )
    
    # 2. Relative Strength Oscillator with Volume Confirmation
    # Compute 14-day RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 5-day/20-day volume ratio
    volume_ratio = data['volume'].rolling(window=5).mean() / data['volume'].rolling(window=20).mean()
    
    # Filter RSI signals using volume ratio thresholds
    rsi_factor = rsi.copy()
    rsi_factor[volume_ratio < 1.1] = 50  # Neutralize signals when volume is low
    rsi_factor = (rsi_factor - 50) / 50  # Normalize to [-1, 1]
    
    # 3. Volatility Regime Adaptive Factor
    # Calculate 10-day average true range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = true_range.rolling(window=10).mean()
    
    # Classify volatility regime
    volatility_regime = atr_10 / data['close'].rolling(window=10).mean()
    high_vol = volatility_regime > volatility_regime.rolling(window=50).quantile(0.7)
    
    # Switch between mean reversion and momentum
    mean_reversion = -data['close'].pct_change(periods=5)
    momentum = data['close'].pct_change(periods=10)
    
    volatility_factor = pd.Series(index=data.index, dtype=float)
    volatility_factor[high_vol] = mean_reversion[high_vol]  # Mean reversion in high vol
    volatility_factor[~high_vol] = momentum[~high_vol]      # Momentum in low vol
    
    # 4. Intraday Pressure Index
    # Calculate buying pressure
    price_range = data['high'] - data['low']
    price_range = price_range.replace(0, 1e-10)  # Avoid division by zero
    intraday_pressure = (data['close'] - data['open']) / price_range
    
    # Accumulate volume-weighted pressure over 5 days
    pressure_weighted = intraday_pressure * data['volume']
    pressure_index = pressure_weighted.rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    
    # 5. Liquidity-Adjusted Price Trend
    # Compute 15-day price trend slope using linear regression
    def linear_slope(x):
        if len(x) < 2:
            return np.nan
        return np.polyfit(range(len(x)), x, 1)[0]
    
    price_trend = data['close'].rolling(window=15).apply(linear_slope, raw=True)
    
    # Calculate liquidity proxy (volume * price / amount)
    liquidity = (data['volume'] * data['close']) / data['amount']
    liquidity = liquidity.replace([np.inf, -np.inf], np.nan)
    
    # Adjust price trend by liquidity
    liquidity_factor = price_trend * (1 / liquidity.rolling(window=15).mean())
    
    # 6. Volume-Cluster Breakout Detector
    # Identify abnormal volume clusters (volume > 2x 20-day average)
    volume_avg_20 = data['volume'].rolling(window=20).mean()
    high_volume = data['volume'] > (2 * volume_avg_20)
    
    # Detect price breakouts during high-volume periods
    price_breakout = (data['close'] - data['close'].rolling(window=5).min()) / data['close'].rolling(window=5).min()
    volume_cluster_factor = pd.Series(0, index=data.index)
    volume_cluster_factor[high_volume] = price_breakout[high_volume]
    
    # 7. Amplitude-Volume Divergence Factor
    # Calculate 10-day price amplitude average
    daily_amplitude = (data['high'] - data['low']) / data['close']
    amplitude_avg = daily_amplitude.rolling(window=10).mean()
    
    # Calculate 10-day volume z-score
    volume_zscore = (data['volume'] - data['volume'].rolling(window=10).mean()) / data['volume'].rolling(window=10).std()
    volume_zscore = volume_zscore.replace([np.inf, -np.inf], np.nan)
    
    # Detect divergences between amplitude and volume patterns
    amplitude_change = amplitude_avg.pct_change(periods=5)
    volume_change = volume_zscore.diff(periods=5)
    
    divergence_factor = amplitude_change - volume_change
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'momentum': momentum_factor,
        'rsi': rsi_factor,
        'volatility': volatility_factor,
        'pressure': pressure_index,
        'liquidity': liquidity_factor,
        'volume_cluster': volume_cluster_factor,
        'divergence': divergence_factor
    })
    
    # Normalize each factor and combine
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x)
    alpha_factor = normalized_factors.mean(axis=1)
    
    return alpha_factor
