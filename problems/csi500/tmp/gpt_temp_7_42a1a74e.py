import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate Dynamic Momentum Quality Factor based on volatility regime detection,
    volume trend analysis, and intraday momentum quality assessment.
    """
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(max(40, len(df))):
        if i < 40:
            continue
            
        current_data = df.iloc[:i+1]
        current_returns = returns.iloc[:i+1]
        
        # Volatility Regime Detection
        ultra_short_vol = current_returns.iloc[i-2:i+1].std() if i >= 2 else np.nan
        short_term_vol = current_returns.iloc[i-5:i+1].std() if i >= 5 else np.nan
        medium_term_vol = current_returns.iloc[i-15:i+1].std() if i >= 15 else np.nan
        
        if pd.isna(ultra_short_vol) or pd.isna(short_term_vol) or short_term_vol == 0:
            continue
            
        volatility_acceleration = (ultra_short_vol - short_term_vol) / short_term_vol
        
        # Adaptive Timeframe Selection
        if volatility_acceleration > 0.25:
            primary_tf, secondary_tf, tertiary_tf = 3, 7, 15
        elif volatility_acceleration >= 0.1:
            primary_tf, secondary_tf, tertiary_tf = 7, 15, 25
        elif volatility_acceleration > -0.1:
            primary_tf, secondary_tf, tertiary_tf = 15, 25, 40
        else:
            primary_tf, secondary_tf, tertiary_tf = 10, 20, 35
        
        # Check if we have enough data for the selected timeframes
        if i < max(primary_tf, secondary_tf, tertiary_tf):
            continue
        
        # Volume Trend Analysis
        current_volume = current_data['volume'].iloc[i]
        
        primary_vol_momentum = current_volume / current_data['volume'].iloc[i-primary_tf] if current_data['volume'].iloc[i-primary_tf] > 0 else 1.0
        secondary_vol_momentum = current_volume / current_data['volume'].iloc[i-secondary_tf] if current_data['volume'].iloc[i-secondary_tf] > 0 else 1.0
        tertiary_vol_momentum = current_volume / current_data['volume'].iloc[i-tertiary_tf] if current_data['volume'].iloc[i-tertiary_tf] > 0 else 1.0
        
        # Volume Trend Classification
        if (primary_vol_momentum > 1.3 and secondary_vol_momentum > 1.2 and tertiary_vol_momentum > 1.1):
            volume_multiplier = 2.0
        elif (primary_vol_momentum > 1.2 and secondary_vol_momentum > 1.1):
            volume_multiplier = 1.5
        elif primary_vol_momentum > 1.1:
            volume_multiplier = 1.2
        elif (primary_vol_momentum < 0.8 and secondary_vol_momentum < 0.85 and tertiary_vol_momentum < 0.9):
            volume_multiplier = 0.6
        elif (primary_vol_momentum < 0.85 and secondary_vol_momentum < 0.9):
            volume_multiplier = 0.8
        elif primary_vol_momentum < 0.9:
            volume_multiplier = 0.9
        else:
            volume_multiplier = 1.0
        
        # Volume-Price Divergence
        current_close = current_data['close'].iloc[i]
        
        price_primary = current_close / current_data['close'].iloc[i-primary_tf] if current_data['close'].iloc[i-primary_tf] > 0 else 1.0
        price_secondary = current_close / current_data['close'].iloc[i-secondary_tf] if current_data['close'].iloc[i-secondary_tf] > 0 else 1.0
        price_tertiary = current_close / current_data['close'].iloc[i-tertiary_tf] if current_data['close'].iloc[i-tertiary_tf] > 0 else 1.0
        
        primary_divergence = price_primary - primary_vol_momentum
        secondary_divergence = price_secondary - secondary_vol_momentum
        tertiary_divergence = price_tertiary - tertiary_vol_momentum
        
        # Intraday Momentum Quality
        current_open = current_data['open'].iloc[i]
        current_high = current_data['high'].iloc[i]
        current_low = current_data['low'].iloc[i]
        prev_close = current_data['close'].iloc[i-1] if i > 0 else current_open
        
        daily_range = current_high - current_low
        if daily_range == 0:
            daily_efficiency = 0
            opening_gap_efficiency = 0
            intraday_range_utilization = 0
        else:
            daily_efficiency = (current_close - current_open) / daily_range
            opening_gap_efficiency = (current_open - prev_close) / daily_range
            intraday_range_utilization = abs(current_close - current_open) / daily_range
        
        # Momentum Quality Assessment
        if (daily_efficiency > 0.6 and opening_gap_efficiency > 0 and intraday_range_utilization > 0.5):
            quality_multiplier = 1.4
        elif (daily_efficiency > 0.3 and opening_gap_efficiency > -0.2):
            quality_multiplier = 1.0
        elif (daily_efficiency < 0.2 or opening_gap_efficiency < -0.3):
            quality_multiplier = 0.7
        elif (daily_efficiency < -0.4 and opening_gap_efficiency > 0.2):
            quality_multiplier = 0.5
        else:
            quality_multiplier = 1.0
        
        # Dynamic Factor Integration
        primary_weight, secondary_weight, tertiary_weight = 0.5, 0.3, 0.2
        base_score = (primary_divergence * primary_weight + 
                     secondary_divergence * secondary_weight + 
                     tertiary_divergence * tertiary_weight)
        
        volume_enhanced_score = base_score * volume_multiplier
        quality_adjusted_score = volume_enhanced_score * quality_multiplier
        
        result.iloc[i] = quality_adjusted_score
    
    return result
