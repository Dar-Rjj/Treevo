import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Divergence Alpha Factor
    Combines multi-timeframe momentum acceleration with volume-price divergence analysis,
    using amount-based regime detection for adaptive signal weighting.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required lookback periods
    max_lookback = 20
    min_periods = max_lookback + 5
    
    if len(df) < min_periods:
        return result
    
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Multi-Timeframe Momentum Components
    momentum_periods = [5, 10, 20]
    
    # Price momentum components
    price_momentum = {}
    for period in momentum_periods:
        price_momentum[period] = (close / close.shift(period)) - 1
    
    # Volume momentum components  
    volume_momentum = {}
    for period in momentum_periods:
        volume_momentum[period] = (volume / volume.shift(period)) - 1
    
    # Momentum Acceleration Calculation
    price_acceleration = {}
    volume_acceleration = {}
    for period in momentum_periods:
        if period > 5:
            price_acceleration[period] = price_momentum[period] - price_momentum[5]
            volume_acceleration[period] = volume_momentum[period] - volume_momentum[5]
        else:
            price_acceleration[period] = price_momentum[period].diff()
            volume_acceleration[period] = volume_momentum[period].diff()
    
    # Volume-Price Divergence Analysis
    divergence_signals = {}
    for period in momentum_periods:
        # Calculate divergence as difference between price and volume momentum
        divergence = price_momentum[period] - volume_momentum[period]
        
        # Classify divergence types
        positive_div = divergence.where(divergence > 0, 0)
        negative_div = divergence.where(divergence < 0, 0)
        convergence = divergence.where((divergence.abs() < 0.01), 0)
        
        # Store divergence strength
        divergence_signals[period] = divergence
    
    # Exponential Smoothing Application
    alpha_smooth = 0.3
    
    # Smooth momentum components
    smoothed_price_momentum = {}
    smoothed_volume_momentum = {}
    for period in momentum_periods:
        smoothed_price_momentum[period] = price_momentum[period].ewm(alpha=alpha_smooth).mean()
        smoothed_volume_momentum[period] = volume_momentum[period].ewm(alpha=alpha_smooth).mean()
    
    # Smooth divergence signals
    smoothed_divergence = {}
    for period in momentum_periods:
        smoothed_divergence[period] = divergence_signals[period].ewm(alpha=alpha_smooth).mean()
    
    # Amount-Based Regime Detection
    amount_20d_avg = amount.rolling(window=20, min_periods=10).mean()
    amount_acceleration = amount_20d_avg.diff() / amount_20d_avg.shift(1)
    
    # Regime classification
    amount_volatility = amount.rolling(window=20, min_periods=10).std()
    regime_threshold = amount_volatility.quantile(0.7)
    
    high_participation_regime = amount_volatility > regime_threshold
    low_participation_regime = amount_volatility <= regime_threshold
    
    # Regime shift detection
    regime_stability = amount_volatility.rolling(window=5).std() / amount_volatility
    transition_periods = regime_stability > regime_stability.quantile(0.8)
    
    # Multi-timeframe Divergence Blending with Regime Weights
    timeframe_weights = {}
    
    for i, date in enumerate(df.index):
        if i < max_lookback:
            continue
            
        # Determine regime-appropriate weights
        if high_participation_regime.iloc[i]:
            # High volatility: emphasize short-term signals
            timeframe_weights[5] = 0.5
            timeframe_weights[10] = 0.3
            timeframe_weights[20] = 0.2
        elif low_participation_regime.iloc[i]:
            # Low volatility: emphasize long-term signals
            timeframe_weights[5] = 0.2
            timeframe_weights[10] = 0.3
            timeframe_weights[20] = 0.5
        else:
            # Default weights
            timeframe_weights[5] = 0.33
            timeframe_weights[10] = 0.33
            timeframe_weights[20] = 0.34
        
        # Adjust for transition periods
        if transition_periods.iloc[i]:
            # Blend weights during transitions
            timeframe_weights[5] = 0.4
            timeframe_weights[10] = 0.4
            timeframe_weights[20] = 0.2
        
        # Calculate weighted divergence signal
        weighted_divergence = 0
        for period in momentum_periods:
            if not pd.isna(smoothed_divergence[period].iloc[i]):
                weighted_divergence += timeframe_weights[period] * smoothed_divergence[period].iloc[i]
        
        # Add momentum acceleration component
        weighted_acceleration = 0
        for period in momentum_periods:
            if not pd.isna(price_acceleration[period].iloc[i]):
                accel_weight = timeframe_weights[period] * 0.5  # Reduce weight for acceleration
                weighted_acceleration += accel_weight * price_acceleration[period].iloc[i]
        
        # Combine divergence and acceleration
        raw_signal = weighted_divergence + weighted_acceleration
        
        # Volatility-Based Scaling
        price_range_20d = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close'].rolling(window=20).mean()
        
        if not pd.isna(price_range_20d.iloc[i]) and price_range_20d.iloc[i] > 0:
            volatility_scaling = 1.0 / price_range_20d.iloc[i]
            # Cap scaling to avoid extreme values
            volatility_scaling = np.clip(volatility_scaling, 0.1, 10.0)
            scaled_signal = raw_signal * volatility_scaling
        else:
            scaled_signal = raw_signal
        
        result.iloc[i] = scaled_signal
    
    # Final smoothing and normalization
    result = result.ewm(alpha=0.1).mean()
    
    # Cross-sectional ranking (placeholder - would be implemented in portfolio context)
    # In practice, this would rank across all stocks in the universe
    
    return result
