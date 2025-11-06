import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Adjusted Divergence factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price Momentum
    # Short-term momentum (1-3 days)
    mom_short_1 = data['close'].pct_change(1)
    mom_short_2 = data['close'].pct_change(2)
    mom_short_3 = data['close'].pct_change(3)
    mom_short = (mom_short_1 + mom_short_2 + mom_short_3) / 3
    
    # Medium-term momentum (5-10 days)
    mom_medium_5 = data['close'].pct_change(5)
    mom_medium_10 = data['close'].pct_change(10)
    mom_medium = (mom_medium_5 + mom_medium_10) / 2
    
    # Long-term momentum (20-50 days)
    mom_long_20 = data['close'].pct_change(20)
    mom_long_50 = data['close'].pct_change(50)
    mom_long = (mom_long_20 + mom_long_50) / 2
    
    # Assess Volatility Conditions
    # Daily volatility using High-Low range
    daily_vol = (data['high'] - data['low']) / data['close']
    
    # Rolling volatility across timeframes
    vol_short = daily_vol.rolling(window=5, min_periods=3).std()
    vol_medium = daily_vol.rolling(window=15, min_periods=10).std()
    vol_long = daily_vol.rolling(window=30, min_periods=20).std()
    
    # Volatility persistence (autocorrelation-like measure)
    vol_persistence = daily_vol.rolling(window=10, min_periods=5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.all(x == x[0]) else 0
    )
    
    # Analyze Volume-Price Relationship
    # Volume trend across timeframes
    volume_trend_short = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_trend_medium = data['volume'].rolling(window=15, min_periods=10).mean()
    volume_trend_long = data['volume'].rolling(window=30, min_periods=20).mean()
    
    # Volume-price divergence detection
    vol_price_div_short = (mom_short.rolling(window=5, min_periods=3).mean() - 
                          data['volume'].pct_change(3).rolling(window=5, min_periods=3).mean())
    vol_price_div_medium = (mom_medium.rolling(window=10, min_periods=7).mean() - 
                           data['volume'].pct_change(10).rolling(window=10, min_periods=7).mean())
    vol_price_div_long = (mom_long.rolling(window=20, min_periods=15).mean() - 
                         data['volume'].pct_change(20).rolling(window=20, min_periods=15).mean())
    
    # Volume confirmation patterns
    volume_confirmation = (
        (data['volume'] > volume_trend_short).astype(int) + 
        (data['volume'] > volume_trend_medium).astype(int) + 
        (data['volume'] > volume_trend_long).astype(int)
    ) / 3
    
    # Generate Composite Signal
    # Adjust momentum by volatility across timeframes
    vol_adj_mom_short = mom_short / (vol_short + 1e-8)
    vol_adj_mom_medium = mom_medium / (vol_medium + 1e-8)
    vol_adj_mom_long = mom_long / (vol_long + 1e-8)
    
    # Detect convergence/divergence patterns
    timeframe_alignment = (
        np.sign(vol_adj_mom_short) * np.sign(vol_adj_mom_medium) * np.sign(vol_adj_mom_long)
    )
    
    # Strength of alignment
    alignment_strength = (
        abs(vol_adj_mom_short) + abs(vol_adj_mom_medium) + abs(vol_adj_mom_long)
    ) / 3
    
    # Volume divergence impact
    volume_divergence_impact = (
        abs(vol_price_div_short) + abs(vol_price_div_medium) + abs(vol_price_div_long)
    ) / 3
    
    # Create final factor
    # Weight volatility-adjusted momentum by volume confirmation
    weighted_momentum = (
        vol_adj_mom_short * volume_confirmation * 0.4 +
        vol_adj_mom_medium * volume_confirmation * 0.35 +
        vol_adj_mom_long * volume_confirmation * 0.25
    )
    
    # Scale by timeframe consistency
    consistency_multiplier = timeframe_alignment * alignment_strength
    
    # Adjust for volatility persistence
    volatility_adjustment = 1 / (1 + abs(vol_persistence))
    
    # Final composite score
    composite_score = (
        weighted_momentum * 
        consistency_multiplier * 
        volatility_adjustment * 
        (1 - volume_divergence_impact)
    )
    
    return composite_score
