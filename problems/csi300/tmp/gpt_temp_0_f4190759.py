import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # Calculate daily range as percentage of close
    daily_range_pct = (data['high'] - data['low']) / data['close']
    
    # Short-term volatility (10-day average)
    short_term_vol = daily_range_pct.rolling(window=10, min_periods=5).mean()
    
    # Long-term volatility (50-day average)
    long_term_vol = daily_range_pct.rolling(window=50, min_periods=25).mean()
    
    # Determine volatility regime
    high_vol_regime = short_term_vol > long_term_vol
    
    # Adaptive Price-Volume Momentum Analysis
    # Price momentum components
    price_momentum_short_hv = data['close'] / data['close'].shift(3) - 1
    price_momentum_long_hv = data['close'] / data['close'].shift(8) - 1
    price_momentum_short_lv = data['close'] / data['close'].shift(5) - 1
    price_momentum_long_lv = data['close'] / data['close'].shift(20) - 1
    
    # Volume momentum components
    volume_momentum_short_hv = data['volume'] / data['volume'].shift(3) - 1
    volume_momentum_long_hv = data['volume'] / data['volume'].shift(8) - 1
    volume_momentum_short_lv = data['volume'] / data['volume'].shift(5) - 1
    volume_momentum_long_lv = data['volume'] / data['volume'].shift(20) - 1
    
    # Select appropriate momentum based on volatility regime
    price_momentum_short = np.where(high_vol_regime, price_momentum_short_hv, price_momentum_short_lv)
    price_momentum_long = np.where(high_vol_regime, price_momentum_long_hv, price_momentum_long_lv)
    volume_momentum_short = np.where(high_vol_regime, volume_momentum_short_hv, volume_momentum_short_lv)
    volume_momentum_long = np.where(high_vol_regime, volume_momentum_long_lv, volume_momentum_long_lv)
    
    # Calculate acceleration metrics
    price_acceleration = price_momentum_short - price_momentum_long
    volume_acceleration = volume_momentum_short - volume_momentum_long
    
    # Divergence Pattern Detection
    positive_divergence = (price_acceleration > 0) & (volume_acceleration < 0)
    negative_divergence = (price_acceleration < 0) & (volume_acceleration > 0)
    
    # Base divergence signal
    base_signal = np.zeros(len(data))
    base_signal[positive_divergence] = 1
    base_signal[negative_divergence] = -1
    
    # Intraday Range Amplification
    current_daily_range = data['high'] - data['low']
    
    # Historical range baseline
    hist_range_window = np.where(high_vol_regime, 5, 10)
    hist_range_baseline = pd.Series(np.nan, index=data.index)
    
    for i in range(len(data)):
        window_size = hist_range_window[i]
        if i >= window_size - 1:
            hist_range_baseline.iloc[i] = current_daily_range.iloc[i-window_size+1:i+1].mean()
    
    # Range multiplier
    range_multiplier = current_daily_range / hist_range_baseline
    range_multiplier = range_multiplier.fillna(1.0)  # Fill NaN with neutral value
    
    # Momentum Quality Assessment
    # Calculate daily returns
    daily_returns = data['close'].pct_change()
    
    # Direction consistency
    lookback_period = np.where(high_vol_regime, 5, 10)
    direction_consistency = pd.Series(np.nan, index=data.index)
    
    for i in range(len(data)):
        period = lookback_period[i]
        if i >= period - 1:
            returns_window = daily_returns.iloc[i-period+1:i+1]
            positive_count = (returns_window > 0).sum()
            direction_consistency.iloc[i] = positive_count / period
    
    direction_consistency = direction_consistency.fillna(0.5)  # Fill NaN with neutral value
    
    # Return stability (inverse of variance)
    return_stability = pd.Series(np.nan, index=data.index)
    
    for i in range(len(data)):
        period = lookback_period[i]
        if i >= period - 1:
            returns_window = daily_returns.iloc[i-period+1:i+1]
            variance = returns_window.var()
            return_stability.iloc[i] = 1 / (1 + variance) if variance > 0 else 1.0
    
    return_stability = return_stability.fillna(1.0)  # Fill NaN with neutral value
    
    # Composite Alpha Generation
    alpha_factor = (base_signal * 
                   range_multiplier * 
                   direction_consistency * 
                   return_stability)
    
    return pd.Series(alpha_factor, index=data.index)
