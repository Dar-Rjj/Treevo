import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multi-timeframe acceleration, regime-adaptive momentum,
    price-volume convergence, efficiency-adjusted momentum, and extreme move confirmation.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    amount = df['amount']
    
    # Multi-Timeframe Acceleration Alignment
    # Price acceleration hierarchy
    ultra_short_price_acc = (close / close.shift(1)) / (close.shift(1) / close.shift(2))
    short_term_price_acc = (close / close.shift(3)) / (close.shift(3) / close.shift(6))
    medium_term_price_acc = (close / close.shift(5)) / (close.shift(5) / close.shift(10))
    
    # Volume acceleration hierarchy
    ultra_short_vol_acc = (volume / volume.shift(1)) / (volume.shift(1) / volume.shift(2))
    short_term_vol_acc = (volume / volume.shift(3)) / (volume.shift(3) / volume.shift(6))
    medium_term_vol_acc = (volume / volume.shift(5)) / (volume.shift(5) / volume.shift(10))
    
    # Amount acceleration hierarchy
    ultra_short_amt_acc = (amount / amount.shift(1)) / (amount.shift(1) / amount.shift(2))
    short_term_amt_acc = (amount / amount.shift(3)) / (amount.shift(3) / amount.shift(6))
    medium_term_amt_acc = (amount / amount.shift(5)) / (amount.shift(5) / amount.shift(10))
    
    # Regime-Adaptive Momentum Signals
    # Volatility regime classification
    range_volatility = (high - low) / close.shift(1)
    gap_volatility = np.abs(open_price - close.shift(1)) / close.shift(1)
    multi_day_volatility = close.diff().abs().rolling(window=5).sum() / close.shift(5)
    
    # Volume regime classification
    volume_intensity = volume / volume.shift(5)
    volume_persistence = volume.rolling(window=5).apply(lambda x: (x > x.shift(1)).sum(), raw=False)
    volume_volatility = np.abs(volume - volume.shift(1)) / volume.shift(1)
    
    # Regime-adaptive factors
    high_vol_momentum = (close / close.shift(3) - 1) * range_volatility
    low_vol_momentum = (close / close.shift(5) - 1) * volume_intensity
    transition_momentum = (close / close.shift(1) - 1) * volume_persistence
    
    # Price-Volume Convergence Divergence
    # Multi-timeframe convergence
    ultra_short_conv = np.sign(close - close.shift(1)) * np.sign(volume - volume.shift(1))
    short_term_conv = np.sign(close - close.shift(3)) * np.sign(volume - volume.shift(3))
    medium_term_conv = np.sign(close - close.shift(5)) * np.sign(volume - volume.shift(5))
    
    # Convergence strength measurement
    convergence_magnitude = np.abs(close - close.shift(1)) * np.abs(volume - volume.shift(1))
    convergence_persistence = ultra_short_conv.rolling(window=5).apply(
        lambda x: (x == x.shift(1)).sum(), raw=False
    )
    multi_scale_conv_alignment = ultra_short_conv * short_term_conv * medium_term_conv
    
    # Divergence transition signals
    convergence_breakdown = convergence_magnitude / convergence_magnitude.shift(3)
    divergence_acceleration = (1 - multi_scale_conv_alignment) / (1 - multi_scale_conv_alignment.shift(3))
    
    # Efficiency-Adjusted Momentum
    # Multi-scale efficiency metrics
    intraday_efficiency = np.abs(close - open_price) / (high - low)
    overnight_efficiency = np.abs(open_price - close.shift(1)) / np.maximum(
        high - low, np.abs(open_price - close.shift(1))
    )
    multi_day_efficiency = np.abs(close - close.shift(3)) / (high - low).rolling(window=3).sum()
    
    # Efficiency-momentum interaction
    high_eff_momentum = (close / close.shift(3) - 1) * intraday_efficiency
    low_eff_momentum = (close / close.shift(5) - 1) * (1 - intraday_efficiency)
    eff_trend_momentum = (close / close.shift(1) - 1) * (intraday_efficiency / intraday_efficiency.shift(3))
    
    # Regime-adaptive efficiency
    vol_scaled_efficiency = intraday_efficiency * range_volatility
    volume_confirmed_efficiency = intraday_efficiency * volume_intensity
    multi_timeframe_efficiency = intraday_efficiency * multi_day_efficiency
    
    # Extreme Move Confirmation
    # Price extreme detection
    intraday_extreme_pos = (close - low) / (high - low)
    short_term_extreme = (close - low.rolling(window=3).min()) / (
        high.rolling(window=3).max() - low.rolling(window=3).min()
    )
    momentum_extreme = (close / close.shift(5) - 1) / (high.shift(5) - low.shift(5)) * close.shift(5)
    
    # Volume confirmation signals
    extreme_vol_confirmation = intraday_extreme_pos * (volume / volume.shift(3))
    vol_acceleration_extreme = momentum_extreme * ultra_short_vol_acc
    multi_timeframe_vol_alignment = intraday_extreme_pos * volume_intensity * volume_persistence
    
    # Regime-adaptive extreme factors
    high_vol_extreme = intraday_extreme_pos * range_volatility
    low_vol_extreme = momentum_extreme * volume_intensity
    transition_extreme = intraday_extreme_pos * convergence_breakdown
    
    # Combine all components into final alpha factor
    # Weight components based on their predictive power and stability
    alpha_factor = (
        # Acceleration alignment (30%)
        0.3 * (
            ultra_short_price_acc.rank(pct=True) + 
            short_term_vol_acc.rank(pct=True) + 
            medium_term_amt_acc.rank(pct=True)
        ) / 3 +
        
        # Regime-adaptive momentum (25%)
        0.25 * (
            high_vol_momentum.rank(pct=True) + 
            low_vol_momentum.rank(pct=True) + 
            transition_momentum.rank(pct=True)
        ) / 3 +
        
        # Price-volume convergence (20%)
        0.2 * (
            multi_scale_conv_alignment.rank(pct=True) + 
            convergence_breakdown.rank(pct=True)
        ) / 2 +
        
        # Efficiency-adjusted momentum (15%)
        0.15 * (
            high_eff_momentum.rank(pct=True) + 
            vol_scaled_efficiency.rank(pct=True)
        ) / 2 +
        
        # Extreme move confirmation (10%)
        0.1 * (
            extreme_vol_confirmation.rank(pct=True) + 
            high_vol_extreme.rank(pct=True)
        ) / 2
    )
    
    return alpha_factor
