import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Price-Volume Divergence with Liquidity Confirmation
    """
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Multi-Timeframe Momentum Divergence Analysis
    # Exponential decay weights
    def exponential_decay_weights(length, decay_rate):
        weights = np.array([decay_rate ** i for i in range(length-1, -1, -1)])
        return weights / weights.sum()
    
    # Short-term momentum (5-day)
    short_weights = exponential_decay_weights(5, 0.9)
    data['short_momentum'] = data['returns'].rolling(window=5).apply(
        lambda x: np.sum(x * short_weights), raw=True
    )
    
    # Medium-term momentum (20-day)
    medium_weights = exponential_decay_weights(20, 0.94)
    data['medium_momentum'] = data['returns'].rolling(window=20).apply(
        lambda x: np.sum(x * medium_weights), raw=True
    )
    
    # Long-term momentum (60-day)
    long_weights = exponential_decay_weights(60, 0.98)
    data['long_momentum'] = data['returns'].rolling(window=60).apply(
        lambda x: np.sum(x * long_weights), raw=True
    )
    
    # Momentum divergence detection
    data['momentum_divergence'] = (
        np.abs(data['short_momentum'] - data['medium_momentum']) +
        np.abs(data['medium_momentum'] - data['long_momentum']) +
        np.abs(data['short_momentum'] - data['long_momentum'])
    ) / 3
    
    # Volatility-adjusted momentum strength
    vol_20d = data['returns'].rolling(window=20).std()
    data['vol_adjusted_strength'] = data['momentum_divergence'] / (vol_20d + 1e-8)
    
    # 2. Volatility Regime Identification
    data['volatility_20d'] = data['returns'].rolling(window=20).std()
    
    # Average True Range (ATR)
    data['tr'] = np.maximum(
        np.maximum(
            data['high'] - data['low'],
            np.abs(data['high'] - data['close'].shift(1))
        ),
        np.abs(data['low'] - data['close'].shift(1))
    )
    data['atr_14d'] = data['tr'].rolling(window=14).mean()
    
    # Volatility regime classification
    vol_median_60d = data['volatility_20d'].rolling(window=60).median()
    data['high_vol_regime'] = (data['volatility_20d'] > vol_median_60d).astype(int)
    
    # Transition periods (rapid vol changes)
    vol_change_5d = data['volatility_20d'] / (data['volatility_20d'].shift(5) + 1e-8)
    data['transition_regime'] = (vol_change_5d > 2) | (vol_change_5d < 0.5)
    
    # 3. Nonlinear Price-Volume Divergence Assessment
    data['price_range'] = data['high'] - data['low']
    data['volume_adjusted_range'] = data['price_range'] * data['volume']
    data['log_volume_range'] = np.log(data['volume_adjusted_range'] + 1)
    
    # Price-volume synchronization
    price_change = data['close'].pct_change()
    volume_change = data['volume'].pct_change()
    data['price_volume_corr'] = price_change.rolling(window=20).corr(volume_change)
    
    # Momentum-volume direction alignment
    data['momentum_volume_alignment'] = (
        np.sign(data['short_momentum']) * np.sign(volume_change)
    )
    
    # Price-volume decoupling
    data['price_volume_decoupling'] = (
        np.abs(data['price_volume_corr']) * 
        (1 - (data['momentum_volume_alignment'] + 1) / 2)
    )
    
    # 4. Liquidity Gap Analysis
    # Bid-ask spread proxy
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    
    # Volume concentration patterns
    data['volume_autocorr'] = data['volume'].rolling(window=10).apply(
        lambda x: pd.Series(x).autocorr(), raw=False
    )
    
    # Volume trend persistence
    volume_increase = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['volume_trend_strength'] = volume_increase.rolling(window=5).sum()
    
    # Abnormal volume clustering
    volume_mean_20d = data['volume'].rolling(window=20).mean()
    volume_std_20d = data['volume'].rolling(window=20).std()
    data['abnormal_volume'] = (
        np.abs(data['volume'] - volume_mean_20d) / (volume_std_20d + 1e-8)
    )
    
    # 5. Regime-Adaptive Signal Generation
    # Base divergence component
    base_divergence = (
        data['vol_adjusted_strength'] * 
        (1 + data['price_volume_decoupling']) *
        data['log_volume_range']
    )
    
    # High volatility regime signals
    high_vol_signal = (
        base_divergence * 
        np.sign(data['short_momentum']) *
        (1 + data['volume_trend_strength'] / 5)
    )
    
    # Low volatility regime signals
    low_vol_signal = (
        base_divergence * 
        (-np.sign(data['short_momentum'])) *
        (1 + data['abnormal_volume'])
    )
    
    # Transition period smoothing
    transition_signal = (
        base_divergence *
        data['volume_autocorr'] *
        np.sign(data['medium_momentum'])
    )
    
    # 6. Composite Alpha Synthesis
    # Regime-adaptive combination
    regime_signal = (
        data['high_vol_regime'] * high_vol_signal +
        (1 - data['high_vol_regime']) * low_vol_signal
    )
    
    # Apply transition smoothing
    final_signal = np.where(
        data['transition_regime'],
        transition_signal,
        regime_signal
    )
    
    # Liquidity confirmation filter
    liquidity_factor = (
        data['spread_proxy'] * 
        (1 + data['abnormal_volume']) *
        (1 - np.abs(data['price_volume_corr']))
    )
    
    # Final alpha factor
    alpha_factor = final_signal * liquidity_factor
    
    return pd.Series(alpha_factor, index=data.index)
