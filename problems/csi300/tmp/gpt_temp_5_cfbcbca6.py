import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Signal Momentum Factor
    Combines price momentum, volume confirmation, and intraday strength
    with volatility regime classification for adaptive signal weighting.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility Regime Classification
    # Daily volatility proxy using high and low prices
    daily_vol = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    
    # Classify volatility regime
    volatility_regime = pd.Series(index=df.index, dtype=str)
    for i in range(60, len(df)):
        window_vol = daily_vol.iloc[i-60:i]
        current_vol = daily_vol.iloc[i]
        
        vol_80th = window_vol.quantile(0.8)
        vol_20th = window_vol.quantile(0.2)
        
        if current_vol > vol_80th:
            volatility_regime.iloc[i] = 'high'
        elif current_vol < vol_20th:
            volatility_regime.iloc[i] = 'low'
        else:
            volatility_regime.iloc[i] = 'normal'
    
    # Price Momentum Calculation
    # Short-term momentum (5-day)
    mom_5d = df['close'] / df['close'].shift(5) - 1
    
    # Medium-term momentum (20-day)
    mom_20d = df['close'] / df['close'].shift(20) - 1
    
    # Momentum quality assessment
    # Momentum consistency over 5 days
    mom_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        recent_mom = [df['close'].iloc[i-j] / df['close'].iloc[i-j-1] - 1 for j in range(5)]
        mom_consistency.iloc[i] = np.std(recent_mom) if len(recent_mom) > 0 else 0
    
    # Momentum acceleration
    mom_acceleration = mom_5d - mom_5d.shift(5)
    
    # Volume Confirmation Signals
    # Volume trend strength
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_trend = df['volume'] / volume_ma_20 - 1
    
    # Volume-price divergence detection
    price_change = df['close'].pct_change()
    volume_change = df['volume'].pct_change()
    volume_price_divergence = np.sign(price_change) != np.sign(volume_change)
    
    # Volume spike identification
    volume_spike = (df['volume'] - volume_ma_20) / volume_ma_20
    volume_acceleration = volume_spike - volume_spike.shift(5)
    
    # Intraday Strength Measurement
    # Daily range efficiency
    daily_range = df['high'] - df['low']
    intraday_move = np.abs(df['close'] - df['open'])
    range_efficiency = intraday_move / daily_range.replace(0, np.nan)
    
    # Intraday momentum persistence
    intraday_direction = np.sign(df['close'] - df['open'])
    intraday_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(1, len(df)):
        if i >= 5:
            recent_directions = intraday_direction.iloc[i-4:i+1]
            intraday_persistence.iloc[i] = len(recent_directions[recent_directions == intraday_direction.iloc[i]]) / 5
        else:
            intraday_persistence.iloc[i] = 0
    
    # Signal Combination and Regime Adjustment
    for i in range(60, len(df)):
        if pd.isna(volatility_regime.iloc[i]):
            continue
            
        regime = volatility_regime.iloc[i]
        
        # Base signals
        momentum_signal = 0.6 * mom_5d.iloc[i] + 0.4 * mom_20d.iloc[i]
        momentum_quality = 1.0 / (1.0 + np.abs(mom_consistency.iloc[i]))
        
        volume_signal = volume_trend.iloc[i] * (1 - np.abs(volume_price_divergence.iloc[i]))
        volume_confirmation = np.tanh(volume_spike.iloc[i]) * (1 + volume_acceleration.iloc[i])
        
        intraday_signal = range_efficiency.iloc[i] * intraday_persistence.iloc[i]
        
        # Regime-specific adjustments
        if regime == 'high':
            # Emphasize mean reversion, reduce momentum weight
            momentum_weight = 0.3
            volume_weight = 0.4
            intraday_weight = 0.3
            volume_threshold_multiplier = 1.5
            
        elif regime == 'low':
            # Emphasize momentum continuation
            momentum_weight = 0.5
            volume_weight = 0.3
            intraday_weight = 0.2
            volume_threshold_multiplier = 0.7
            
        else:  # normal regime
            momentum_weight = 0.4
            volume_weight = 0.35
            intraday_weight = 0.25
            volume_threshold_multiplier = 1.0
        
        # Apply volume confirmation threshold
        volume_confirmation_adj = volume_confirmation * (np.abs(volume_confirmation) > 0.1 * volume_threshold_multiplier)
        
        # Combine signals with regime-appropriate weights
        factor_value = (
            momentum_weight * momentum_signal * momentum_quality +
            volume_weight * volume_signal * volume_confirmation_adj +
            intraday_weight * intraday_signal
        )
        
        # Risk Management Components
        # Extreme value detection and smoothing
        if i >= 20:
            recent_values = result.iloc[i-19:i]
            if len(recent_values) > 0:
                recent_mean = recent_values.mean()
                recent_std = recent_values.std()
                if recent_std > 0:
                    z_score = (factor_value - recent_mean) / recent_std
                    if np.abs(z_score) > 3:  # extreme value
                        factor_value = np.sign(factor_value) * min(np.abs(factor_value), recent_mean + 2 * recent_std)
        
        # Signal persistence filter
        if i >= 5:
            recent_signals = np.sign([result.iloc[i-j] for j in range(1, min(6, i+1)) if not pd.isna(result.iloc[i-j])])
            if len(recent_signals) >= 3:
                signal_consistency = np.mean(recent_signals == np.sign(factor_value))
                if signal_consistency < 0.6:  # weak persistence
                    factor_value *= 0.7  # decay stale signals
        
        result.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
