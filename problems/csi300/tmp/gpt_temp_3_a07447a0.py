import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Calculate rolling returns with exponential decay weighting
    returns_5 = data['close'].pct_change(5)
    returns_10 = data['close'].pct_change(10)
    returns_20 = data['close'].pct_change(20)
    
    # Exponential decay weights (more weight to recent periods)
    decay_weights_5 = np.exp(-np.arange(5)/2)[::-1]
    decay_weights_10 = np.exp(-np.arange(10)/3)[::-1]
    decay_weights_20 = np.exp(-np.arange(20)/5)[::-1]
    
    # Apply decay weighting to returns
    decayed_momentum = (
        0.4 * returns_5.rolling(5).apply(lambda x: np.sum(x * decay_weights_5) if len(x) == 5 else np.nan) +
        0.35 * returns_10.rolling(10).apply(lambda x: np.sum(x * decay_weights_10) if len(x) == 10 else np.nan) +
        0.25 * returns_20.rolling(20).apply(lambda x: np.sum(x * decay_weights_20) if len(x) == 20 else np.nan)
    )
    
    # Compute momentum acceleration (rate of change of decayed momentum)
    momentum_acceleration = decayed_momentum.diff(3) / decayed_momentum.rolling(5).std()
    
    # Dynamic Volatility Regime Detection
    # Calculate volatility metrics
    data['vol_5_day'] = (data['high'] - data['low']) / data['close']
    
    # 10-day volatility using rolling max/min
    high_10d = data['high'].rolling(5).max()
    low_10d = data['low'].rolling(5).min()
    data['vol_10_day'] = (high_10d - low_10d) / data['close']
    
    # Volatility ratio
    vol_ratio = data['vol_5_day'] / data['vol_10_day']
    
    # Classify market conditions
    high_vol_regime = vol_ratio > 1.5
    normal_vol_regime = (vol_ratio >= 0.8) & (vol_ratio <= 1.5)
    low_vol_regime = vol_ratio < 0.8
    
    # Price-Volume Efficiency Analysis
    # Calculate efficiency metrics
    price_efficiency = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    volume_5d_sum = data['volume'].rolling(5).sum()
    volume_concentration = data['volume'] / volume_5d_sum
    
    # Volume acceleration (ROC of volume)
    vol_roc_1 = data['volume'].pct_change(1)
    vol_roc_3 = data['volume'].pct_change(3)
    vol_roc_5 = data['volume'].pct_change(5)
    volume_acceleration = (vol_roc_1 + vol_roc_3 + vol_roc_5) / 3
    
    # Detect efficiency-strength patterns
    strong_efficiency = (abs(price_efficiency) > 0.7) & (volume_concentration > 0.3)
    medium_efficiency = (abs(price_efficiency) > 0.3) & (volume_acceleration > 0)
    weak_efficiency = ~(strong_efficiency | medium_efficiency)
    
    # Assign efficiency strength scores
    efficiency_strength = np.zeros(len(data))
    efficiency_strength[strong_efficiency] = 1.0
    efficiency_strength[medium_efficiency] = 0.5
    efficiency_strength[weak_efficiency] = 0.1
    
    # Volume-Price Divergence Assessment
    # Calculate Volume Metrics
    short_term_vol_mean = data['volume'].rolling(5).mean()
    medium_term_vol_mean = data['volume'].rolling(10).mean()
    volume_divergence_ratio = short_term_vol_mean / medium_term_vol_mean
    
    # Analyze Price-Volume Relationship
    effective_price = data['amount'] / data['volume'].replace(0, np.nan)
    price_volume_divergence = abs((data['close'] - effective_price) / data['close'])
    
    # Detect divergence patterns
    price_up = data['close'] > data['open']
    volume_down = data['volume'] < data['volume'].shift(1)
    bullish_divergence = price_up & volume_down & (price_volume_divergence < price_volume_divergence.rolling(10).quantile(0.3))
    
    price_down = data['close'] < data['open']
    volume_up = data['volume'] > data['volume'].shift(1)
    bearish_divergence = price_down & volume_up & (price_volume_divergence > price_volume_divergence.rolling(10).quantile(0.7))
    
    confirmed_movement = ~(bullish_divergence | bearish_divergence)
    
    # Assign divergence scores
    divergence_score = np.zeros(len(data))
    divergence_score[bullish_divergence] = 1.0
    divergence_score[bearish_divergence] = -1.0
    divergence_score[confirmed_movement] = 0.0
    
    # Regime-Adaptive Signal Synthesis
    signal = np.zeros(len(data))
    
    # High volatility regime weights
    high_vol_signal = (
        0.4 * momentum_acceleration * efficiency_strength +
        0.3 * divergence_score +
        0.2 * volume_divergence_ratio +
        0.1 * volume_concentration
    )
    
    # Normal volatility regime weights
    normal_vol_signal = (
        0.3 * momentum_acceleration * efficiency_strength +
        0.3 * divergence_score +
        0.2 * volume_divergence_ratio +
        0.2 * volume_concentration
    )
    
    # Low volatility regime weights
    low_vol_signal = (
        0.2 * momentum_acceleration * efficiency_strength +
        0.4 * divergence_score +
        0.2 * volume_divergence_ratio +
        0.2 * volume_concentration
    )
    
    # Apply regime-specific signals
    signal[high_vol_regime] = high_vol_signal[high_vol_regime]
    signal[normal_vol_regime] = normal_vol_signal[normal_vol_regime]
    signal[low_vol_regime] = low_vol_signal[low_vol_regime]
    
    # Apply Directional Logic
    directional_multiplier = np.where(price_efficiency > 0, 1.0, -1.0)
    signal = signal * directional_multiplier
    
    # Scale by volume trend strength
    volume_trend = data['volume'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan)
    signal = signal * (1 + 0.5 * np.tanh(volume_trend / volume_trend.rolling(10).std()))
    
    # Final Signal Processing
    # Scale by current 5-day volatility
    signal = signal * data['vol_5_day']
    
    # Apply cubic root transformation for signal stability
    signal = np.sign(signal) * np.power(abs(signal), 1/3)
    
    # Return as pandas Series
    return pd.Series(signal, index=data.index)
