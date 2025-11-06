import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate Multi-Timeframe Momentum with Decay
    def exponential_decay_momentum(close, window):
        weights = np.exp(-np.arange(window) / (window / 3))
        weights = weights / weights.sum()
        
        momentum = np.zeros(len(close))
        for i in range(window, len(close)):
            period_data = close.iloc[i-window:i]
            momentum[i] = (close.iloc[i] - np.average(period_data, weights=weights)) / np.average(period_data, weights=weights)
        
        return pd.Series(momentum, index=close.index)
    
    # Short-term momentum (1-3 days)
    mom_short = exponential_decay_momentum(data['close'], 3)
    
    # Medium-term momentum (5-10 days)
    mom_medium = exponential_decay_momentum(data['close'], 10)
    
    # Long-term momentum (20-50 days)
    mom_long = exponential_decay_momentum(data['close'], 50)
    
    # Detect Momentum Convergence Patterns
    def calculate_momentum_spreads(short, medium, long):
        spread_sm = short - medium
        spread_ml = medium - long
        spread_sl = short - long
        
        # Absolute differences
        abs_spread_sm = abs(spread_sm)
        abs_spread_ml = abs(spread_ml)
        abs_spread_sl = abs(spread_sl)
        
        return spread_sm, spread_ml, spread_sl, abs_spread_sm, abs_spread_ml, abs_spread_sl
    
    spread_sm, spread_ml, spread_sl, abs_spread_sm, abs_spread_ml, abs_spread_sl = calculate_momentum_spreads(
        mom_short, mom_medium, mom_long
    )
    
    # Identify Convergence Signals
    def momentum_convergence_signal(abs_spread_sm, abs_spread_ml, abs_spread_sl, short, medium, long):
        convergence = np.zeros(len(abs_spread_sm))
        
        for i in range(5, len(abs_spread_sm)):
            # Check for decreasing spreads over recent periods
            recent_decrease_sm = (abs_spread_sm.iloc[i] < abs_spread_sm.iloc[i-3:i].mean())
            recent_decrease_ml = (abs_spread_ml.iloc[i] < abs_spread_ml.iloc[i-3:i].mean())
            recent_decrease_sl = (abs_spread_sl.iloc[i] < abs_spread_sl.iloc[i-3:i].mean())
            
            # Weight by recent performance (all momenta positive)
            recent_perf_weight = np.sign(short.iloc[i]) * np.sign(medium.iloc[i]) * np.sign(long.iloc[i])
            
            if recent_decrease_sm and recent_decrease_ml and recent_decrease_sl:
                convergence[i] = recent_perf_weight * (
                    (1 - abs_spread_sm.iloc[i]) + 
                    (1 - abs_spread_ml.iloc[i]) + 
                    (1 - abs_spread_sl.iloc[i])
                ) / 3
        
        return pd.Series(convergence, index=abs_spread_sm.index)
    
    convergence_signal = momentum_convergence_signal(
        abs_spread_sm, abs_spread_ml, abs_spread_sl, mom_short, mom_medium, mom_long
    )
    
    # Calculate Volatility Regime
    def true_range_volatility(high, low, close):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Rolling volatility (20-day)
        vol_rolling = true_range.rolling(window=20, min_periods=10).std()
        return vol_rolling
    
    volatility = true_range_volatility(data['high'], data['low'], data['close'])
    
    def classify_volatility_regime(volatility, window=63):
        vol_percentile = volatility.rolling(window=window, min_periods=30).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # High volatility: above 0.5 std, Low volatility: below -0.5 std
        vol_regime = pd.Series(0, index=volatility.index)
        vol_regime[vol_percentile > 0.5] = 1  # High volatility
        vol_regime[vol_percentile < -0.5] = -1  # Low volatility
        
        return vol_regime
    
    vol_regime = classify_volatility_regime(volatility)
    
    # Calculate Volume Acceleration Profile
    def volume_acceleration_profile(volume, windows=[1, 3, 5]):
        volume_roc = {}
        for window in windows:
            volume_roc[window] = (volume / volume.shift(window) - 1).fillna(0)
        
        # Volume acceleration (second difference)
        accel_3d = volume_roc[3] - volume_roc[1]
        accel_5d = volume_roc[5] - volume_roc[3]
        
        # Combined acceleration profile
        volume_accel = (accel_3d + accel_5d) / 2
        return volume_accel
    
    volume_accel = volume_acceleration_profile(data['volume'])
    
    # Combine All Components
    def combine_components(convergence, vol_regime, volume_accel):
        # Scale momentum convergence by volatility regime
        vol_scaling = np.where(vol_regime == 1, 0.7,  # High vol: reduce
                              np.where(vol_regime == -1, 1.3, 1.0))  # Low vol: enhance
        
        scaled_convergence = convergence * vol_scaling
        
        # Multiply by volume acceleration confirmation
        volume_weight = np.tanh(volume_accel * 2)  # Scale volume acceleration
        combined_signal = scaled_convergence * (1 + volume_weight)
        
        # Apply non-linear transformation
        final_signal = np.tanh(combined_signal * 0.5)  # Control for extreme conditions
        
        return pd.Series(final_signal, index=convergence.index)
    
    factor = combine_components(convergence_signal, vol_regime, volume_accel)
    
    return factor
