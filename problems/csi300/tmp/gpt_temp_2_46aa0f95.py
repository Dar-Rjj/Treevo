import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Range Momentum Divergence
    # Calculate short-term high-low range slope (t-4 to t)
    data['high_low_range'] = data['high'] - data['low']
    data['short_term_range_slope'] = (data['high_low_range'] - data['high_low_range'].shift(4)) / 4
    
    # Calculate medium-term high-low range slope (t-9 to t-5)
    data['medium_term_range_slope'] = (data['high_low_range'].shift(5) - data['high_low_range'].shift(9)) / 4
    
    # Volume trend correlation (t-4 to t)
    volume_corr = data['volume'].rolling(window=5).corr(data['high_low_range'])
    
    # Momentum divergence signal
    momentum_divergence = (data['short_term_range_slope'] - data['medium_term_range_slope']) * volume_corr
    
    # Volume-Weighted Price Acceleration
    # Calculate volume-weighted returns (t-2 to t)
    data['vwap'] = (data['close'] * data['volume']).rolling(window=3).sum() / data['volume'].rolling(window=3).sum()
    data['weighted_return'] = data['vwap'].pct_change(periods=2)
    
    # Second-order difference of weighted returns
    data['acceleration'] = data['weighted_return'].diff().diff()
    
    # Sign and magnitude consistency
    sign_consistency = np.sign(data['weighted_return'].rolling(window=3).apply(
        lambda x: 1 if len(set(np.sign(x))) == 1 else 0, raw=True
    ))
    acceleration_factor = data['acceleration'] * sign_consistency
    
    # Intraday Range Efficiency Factor
    # Calculate daily range utilization
    data['close_return'] = data['close'].pct_change()
    data['range_efficiency'] = abs(data['close_return']) / (data['high_low_range'] / data['close'].shift(1))
    
    # Historical range percentile (t-20 to t-1)
    historical_percentile = data['range_efficiency'].shift(1).rolling(window=20).rank(pct=True)
    
    # Multi-day efficiency pattern (t-2 to t)
    efficiency_trend = data['range_efficiency'].rolling(window=3).apply(
        lambda x: 1 if (x.iloc[-1] > x.iloc[0]) and (x.iloc[-1] > x.iloc[1]) else -1, raw=False
    )
    efficiency_factor = data['range_efficiency'] * historical_percentile * efficiency_trend
    
    # Price-Volume Convergence Oscillator
    # Calculate price and volume momentum vectors (t-4 to t)
    price_momentum = data['close'].pct_change(periods=4)
    volume_momentum = data['volume'].pct_change(periods=4)
    
    # Dot product measurement
    convergence_oscillator = price_momentum * volume_momentum
    
    # Extreme value detection
    extreme_threshold = convergence_oscillator.rolling(window=20).quantile(0.9)
    convergence_factor = convergence_oscillator * (convergence_oscillator > extreme_threshold)
    
    # Range Breakout Confirmation Factor
    # Price vs recent high/low (t-5 to t-1)
    recent_high = data['high'].shift(1).rolling(window=5).max()
    recent_low = data['low'].shift(1).rolling(window=5).min()
    
    breakout_signal = np.where(
        data['close'] > recent_high, 1,
        np.where(data['close'] < recent_low, -1, 0)
    )
    
    # Volume surge vs historical mean (t-10 to t-1)
    volume_mean = data['volume'].shift(1).rolling(window=10).mean()
    volume_surge = (data['volume'] > volume_mean * 1.2).astype(int)
    
    # Post-breakout returns and range utilization validation
    breakout_confirmation = breakout_signal * volume_surge * data['range_efficiency']
    
    # Volatility-Regime Adaptive Momentum
    # Classify volatility regime (High-Low Range Percentile, t-10 to t-1)
    vol_percentile = data['high_low_range'].shift(1).rolling(window=10).rank(pct=True)
    vol_regime = np.where(vol_percentile > 0.7, 'high', 
                         np.where(vol_percentile < 0.3, 'low', 'normal'))
    
    # Adaptive lookback period based on volatility regime
    def adaptive_momentum(row):
        if row.name in data.index:
            idx = data.index.get_loc(row.name)
            if vol_regime[idx] == 'high':
                lookback = 3
            elif vol_regime[idx] == 'low':
                lookback = 10
            else:
                lookback = 5
            
            if idx >= lookback:
                return data['close'].iloc[idx] / data['close'].iloc[idx - lookback] - 1
        return np.nan
    
    adaptive_momentum_values = pd.Series(data.index, index=data.index).apply(adaptive_momentum)
    
    # Volume-confirmed signals
    volume_confirmation = data['volume'].pct_change(periods=5) > 0
    regime_momentum = adaptive_momentum_values * volume_confirmation
    
    # Combine all factors with equal weights
    combined_factor = (
        momentum_divergence.fillna(0) +
        acceleration_factor.fillna(0) +
        efficiency_factor.fillna(0) +
        convergence_factor.fillna(0) +
        breakout_confirmation.fillna(0) +
        regime_momentum.fillna(0)
    ) / 6
    
    return combined_factor
