import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Regime-Aware Price Amplitude
    # Compute rolling price range efficiency
    window_regime = 20
    high_roll_max = data['high'].rolling(window=window_regime, min_periods=1).max()
    low_roll_min = data['low'].rolling(window=window_regime, min_periods=1).min()
    price_range_efficiency = (data['high'] - data['low']) / (high_roll_max - low_roll_min).replace(0, np.nan)
    price_range_efficiency = price_range_efficiency.fillna(0)
    
    # Identify regime transitions using rolling volatility
    volatility_rolling = data['close'].pct_change().rolling(window=window_regime, min_periods=1).std()
    volatility_regime = (volatility_rolling > volatility_rolling.rolling(window=60, min_periods=1).median()).astype(int)
    
    # Adjust amplitude by regime
    regime_adjusted_amplitude = price_range_efficiency * (1 + 0.5 * volatility_regime)
    
    # 2. Integrate Volume-Weighted Momentum
    # Compute volume-accelerated returns with time-decay weighting
    momentum_window = 10
    volume_weighted_returns = []
    
    for i in range(len(data)):
        if i < momentum_window:
            volume_weighted_returns.append(0)
            continue
            
        weights = np.exp(np.linspace(-1, 0, momentum_window))
        weights = weights / weights.sum()
        
        period_returns = []
        for j in range(momentum_window):
            idx = i - j
            if idx > 0:
                ret = (data['close'].iloc[idx] - data['close'].iloc[idx-1]) / data['close'].iloc[idx-1]
                vol_weight = data['volume'].iloc[idx] / (data['volume'].iloc[idx-momentum_window:idx].mean() + 1e-8)
                period_returns.append(ret * vol_weight * weights[j])
        
        volume_weighted_returns.append(sum(period_returns))
    
    volume_momentum = pd.Series(volume_weighted_returns, index=data.index)
    
    # Combine with amplitude signals
    amplitude_momentum = regime_adjusted_amplitude * volume_momentum
    
    # 3. Apply Volume-Confirmed Persistence Filter
    # Compute open-to-close efficiency
    intraday_efficiency = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    intraday_efficiency = intraday_efficiency.fillna(0)
    
    # Measure persistence across regimes
    persistence_high_vol = intraday_efficiency.rolling(window=5, min_periods=1).std()
    persistence_low_vol = intraday_efficiency.rolling(window=10, min_periods=1).std()
    
    regime_persistence = np.where(volatility_regime == 1, 
                                1 / (1 + persistence_high_vol),
                                1 / (1 + persistence_low_vol))
    
    # Compute volume acceleration
    volume_acceleration = data['volume'].pct_change(periods=3).rolling(window=5, min_periods=1).mean()
    volume_acceleration = volume_acceleration.fillna(0)
    
    # Final factor combination
    final_factor = amplitude_momentum * regime_persistence * (1 + volume_acceleration)
    
    return final_factor
