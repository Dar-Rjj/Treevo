import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Acceleration Calculation
    # Short-Term Acceleration: (Close[t]/Close[t-3] - 1) - (Close[t-3]/Close[t-6] - 1)
    short_term_accel = (data['close'] / data['close'].shift(3) - 1) - (data['close'].shift(3) / data['close'].shift(6) - 1)
    
    # Medium-Term Acceleration: (Close[t]/Close[t-10] - 1) - (Close[t-10]/Close[t-20] - 1)
    medium_term_accel = (data['close'] / data['close'].shift(10) - 1) - (data['close'].shift(10) / data['close'].shift(20) - 1)
    
    # Acceleration Divergence: Short-Term Acceleration - Medium-Term Acceleration
    accel_divergence = short_term_accel - medium_term_accel
    
    # Volume-Weighted Price Analysis
    # Calculate rolling volume-weighted price (5-day window)
    def calculate_vwp(window):
        close_prices = window['close']
        volumes = window['volume']
        return (close_prices * volumes).sum() / volumes.sum() if volumes.sum() != 0 else 0
    
    # Create rolling windows for VWP calculation
    vwp_series = []
    for i in range(len(data)):
        if i >= 4:  # Need at least 5 days of data
            window_data = data.iloc[i-4:i+1]  # Current day + previous 4 days
            vwp = calculate_vwp(window_data)
            vwp_series.append(vwp)
        else:
            vwp_series.append(float('nan'))
    
    data['vwp'] = vwp_series
    
    # VWP Momentum: VWP[t] / VWP[t-5] - 1
    vwp_momentum = data['vwp'] / data['vwp'].shift(5) - 1
    
    # Price-VWP Divergence: (Close[t]/Close[t-5] - 1) - VWP Momentum
    price_vwp_divergence = (data['close'] / data['close'].shift(5) - 1) - vwp_momentum
    
    # Adaptive Factor Construction
    # Calculate 20-day volume moving average
    volume_ma_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Volume regimes
    high_volume_regime = data['volume'] > volume_ma_20
    low_volume_regime = data['volume'] < volume_ma_20
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Regime-specific factors
    # High Volume: Acceleration Divergence × Price-VWP Divergence
    factor[high_volume_regime] = accel_divergence[high_volume_regime] * price_vwp_divergence[high_volume_regime]
    
    # Low Volume: Medium-Term Acceleration × Price-VWP Divergence
    factor[low_volume_regime] = medium_term_accel[low_volume_regime] * price_vwp_divergence[low_volume_regime]
    
    # For neutral volume (equal to average), use weighted average
    neutral_regime = ~high_volume_regime & ~low_volume_regime
    if neutral_regime.any():
        factor[neutral_regime] = (0.5 * accel_divergence[neutral_regime] * price_vwp_divergence[neutral_regime] + 
                                 0.5 * medium_term_accel[neutral_regime] * price_vwp_divergence[neutral_regime])
    
    return factor
