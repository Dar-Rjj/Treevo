import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor using multi-timeframe momentum divergence,
    volatility-regime adaptive weighting, volume anomaly detection, intraday price efficiency,
    and multi-level support/resistance framework.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate returns for volatility calculation
    returns = df['close'].pct_change()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate all components for each date
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Momentum Divergence
        divergences = {}
        
        # Ultra-Short-Term (2-Day)
        if i >= 2:
            price_momentum_2d = df.iloc[i]['close'] / df.iloc[i-2]['close']
            volume_momentum_2d = df.iloc[i]['volume'] / df.iloc[i-2]['volume']
            divergences['ultra_short'] = price_momentum_2d - volume_momentum_2d
        
        # Short-Term (5-Day)
        if i >= 5:
            price_momentum_5d = df.iloc[i]['close'] / df.iloc[i-5]['close']
            volume_momentum_5d = df.iloc[i]['volume'] / df.iloc[i-5]['volume']
            divergences['short'] = price_momentum_5d - volume_momentum_5d
        
        # Medium-Term (10-Day)
        if i >= 10:
            price_momentum_10d = df.iloc[i]['close'] / df.iloc[i-10]['close']
            volume_momentum_10d = df.iloc[i]['volume'] / df.iloc[i-10]['volume']
            divergences['medium'] = price_momentum_10d - volume_momentum_10d
        
        # Long-Term (20-Day)
        if i >= 20:
            price_momentum_20d = df.iloc[i]['close'] / df.iloc[i-20]['close']
            volume_momentum_20d = df.iloc[i]['volume'] / df.iloc[i-20]['volume']
            divergences['long'] = price_momentum_20d - volume_momentum_20d
        
        # Volatility-Regime Adaptive Weighting
        vol_weights = {}
        
        # Volatility Calculation
        ultra_short_vol = returns.iloc[i-1:i+1].std() if i >= 1 else np.nan
        short_term_vol = returns.iloc[i-4:i+1].std() if i >= 4 else np.nan
        medium_term_vol = returns.iloc[i-9:i+1].std() if i >= 9 else np.nan
        long_term_vol = returns.iloc[i-19:i+1].std() if i >= 19 else np.nan
        
        # Regime-Based Weighting Scheme
        if not np.isnan(ultra_short_vol) and not np.isnan(medium_term_vol):
            vol_ratio = ultra_short_vol / medium_term_vol
            
            if vol_ratio > 2.5:
                # Extreme Volatility
                vol_weights = {'ultra_short': 0.7, 'short': 0.3, 'medium': 0.0, 'long': 0.0}
            elif vol_ratio > 1.8:
                # High Volatility
                vol_weights = {'ultra_short': 0.5, 'short': 0.4, 'medium': 0.1, 'long': 0.0}
            elif vol_ratio < 0.6:
                # Low Volatility
                vol_weights = {'ultra_short': 0.1, 'short': 0.2, 'medium': 0.3, 'long': 0.4}
            else:
                # Normal Volatility
                vol_weights = {'ultra_short': 0.2, 'short': 0.3, 'medium': 0.3, 'long': 0.2}
        else:
            # Default to normal volatility if insufficient data
            vol_weights = {'ultra_short': 0.2, 'short': 0.3, 'medium': 0.3, 'long': 0.2}
        
        # Volume Anomaly Detection
        volume_multiplier = 1.0
        
        if i >= 15:
            # Volume Baseline
            recent_volumes = df['volume'].iloc[i-14:i+1]
            volume_median = recent_volumes.median()
            volume_mad = (recent_volumes - volume_median).abs().median()
            current_volume = df.iloc[i]['volume']
            
            # Anomaly Multiplier
            if volume_mad > 0:
                deviation = (current_volume - volume_median) / volume_mad
                
                if deviation > 3.5:
                    volume_multiplier = 3.0
                elif deviation > 2.5:
                    volume_multiplier = 2.2
                elif deviation > 1.5:
                    volume_multiplier = 1.6
        
        # Intraday Price Efficiency
        efficiency_score = 1.0
        
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        current_close = df.iloc[i]['close']
        
        if current_high != current_low:
            normalized_range = (current_high - current_low) / current_close
            close_position = (current_close - current_low) / (current_high - current_low)
            
            if close_position > 0.8 and normalized_range > 0.03:
                efficiency_score = 1.5
            elif close_position > 0.65 and normalized_range > 0.02:
                efficiency_score = 1.25
            elif close_position < 0.35 or normalized_range < 0.01:
                efficiency_score = 0.7
        
        # Multi-Level Support/Resistance Framework
        position_multiplier = 1.0
        
        if i >= 30:
            # Key Levels Identification
            high_10d = df['high'].iloc[i-9:i+1].max()
            low_10d = df['low'].iloc[i-9:i+1].min()
            high_20d = df['high'].iloc[i-19:i+1].max()
            low_20d = df['low'].iloc[i-19:i+1].min()
            high_30d = df['high'].iloc[i-29:i+1].max()
            low_30d = df['low'].iloc[i-29:i+1].min()
            
            current_close = df.iloc[i]['close']
            
            # Position Multiplier
            if (current_close > 0.99 * high_10d and 
                current_close > 0.97 * high_20d and 
                current_close > 0.95 * high_30d):
                position_multiplier = 0.6
            elif (current_close > 0.97 * high_10d and 
                  current_close > 0.95 * high_20d):
                position_multiplier = 0.75
            elif current_close > 0.95 * high_10d:
                position_multiplier = 0.9
            elif (current_close < 1.01 * low_10d and 
                  current_close < 1.03 * low_20d and 
                  current_close < 1.05 * low_30d):
                position_multiplier = 1.4
            elif (current_close < 1.03 * low_10d and 
                  current_close < 1.05 * low_20d):
                position_multiplier = 1.25
            elif current_close < 1.05 * low_10d:
                position_multiplier = 1.1
        
        # Composite Alpha Construction
        weighted_divergence = 0.0
        
        for timeframe, weight in vol_weights.items():
            if timeframe in divergences:
                weighted_divergence += divergences[timeframe] * weight
        
        # Apply all adjustments
        volume_enhanced = weighted_divergence * volume_multiplier
        efficiency_adjusted = volume_enhanced * efficiency_score
        final_alpha = efficiency_adjusted * position_multiplier
        
        result.iloc[i] = final_alpha
    
    return result
