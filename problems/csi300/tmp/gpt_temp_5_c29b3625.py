import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Asymmetric Efficiency Momentum factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    # Calculate rolling windows
    for i in range(len(df)):
        if i < 5:  # Need at least 5 days for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Ultra-Short Efficiency (2-day)
        if i >= 2:
            ultra_high = max(current_data['high'].iloc[i-1:i+1])
            ultra_low = min(current_data['low'].iloc[i-1:i+1])
            ultra_true_range = ultra_high - ultra_low
            ultra_price_movement = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-2])
            ultra_efficiency = ultra_price_movement / ultra_true_range if ultra_true_range > 0 else 0
        else:
            ultra_efficiency = 0
        
        # Short-Term Efficiency (5-day)
        short_high = max(current_data['high'].iloc[i-4:i+1])
        short_low = min(current_data['low'].iloc[i-4:i+1])
        short_true_range = short_high - short_low
        short_price_movement = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-5])
        short_efficiency = short_price_movement / short_true_range if short_true_range > 0 else 0
        
        # Efficiency acceleration gap
        efficiency_acceleration = ultra_efficiency - short_efficiency
        
        # Opening Auction Asymmetry
        opening_gap_momentum = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1] if current_data['close'].iloc[i-1] > 0 else 0
        auction_imbalance = (current_data['open'].iloc[i] - current_data['low'].iloc[i]) - (current_data['high'].iloc[i] - current_data['open'].iloc[i])
        
        # Gap absorption efficiency
        high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if high_low_range > 0 and current_data['close'].iloc[i-1] > 0:
            gap_absorption = (current_data['open'].iloc[i] / current_data['close'].iloc[i-1]) * ((current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range)
        else:
            gap_absorption = 0
        
        # Intraday Directional Bias
        intraday_bias = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range if high_low_range > 0 else 0
        
        # Closing pressure asymmetry
        if high_low_range > 0:
            closing_pressure = ((current_data['close'].iloc[i] - current_data['low'].iloc[i]) / high_low_range) - ((current_data['high'].iloc[i] - current_data['close'].iloc[i]) / high_low_range)
        else:
            closing_pressure = 0
        
        # Volume Asymmetry Analysis (5-day)
        up_volume_sum = 0
        down_volume_sum = 0
        for j in range(max(0, i-4), i+1):
            if current_data['close'].iloc[j] > current_data['open'].iloc[j]:
                up_volume_sum += current_data['volume'].iloc[j]
            elif current_data['close'].iloc[j] < current_data['open'].iloc[j]:
                down_volume_sum += current_data['volume'].iloc[j]
        
        volume_asymmetry = up_volume_sum / down_volume_sum if down_volume_sum > 0 else 1
        
        # Volatility skew ratio
        if high_low_range > 0:
            volatility_skew = ((current_data['high'].iloc[i] - current_data['open'].iloc[i]) / high_low_range) - ((current_data['open'].iloc[i] - current_data['low'].iloc[i]) / high_low_range)
        else:
            volatility_skew = 0
        
        # Core Cross-Asset Efficiency Signal
        relative_efficiency_momentum = short_efficiency
        base_signal = relative_efficiency_momentum * efficiency_acceleration
        
        # Microstructure Adjusted
        microstructure_adjusted = base_signal * gap_absorption
        
        # Liquidity Enhanced
        liquidity_enhanced = microstructure_adjusted * volume_asymmetry
        
        # Asymmetric Volatility Integration
        volatility_asymmetry_score = volatility_skew
        
        # Microstructure Momentum Components
        microstructure_momentum = (opening_gap_momentum + intraday_bias + closing_pressure) / 3
        
        # Liquidity-Weighted Enhancement
        liquidity_weighted = volume_asymmetry
        
        # Cross-Asset Divergence Multiplier (simplified as combination of signals)
        divergence_multiplier = (efficiency_acceleration + microstructure_momentum + volatility_asymmetry_score) / 3
        
        # Final Composite Factor
        composite_factor = (
            liquidity_enhanced * 0.4 +
            volatility_asymmetry_score * 0.2 +
            microstructure_momentum * 0.2 +
            liquidity_weighted * 0.1 +
            divergence_multiplier * 0.1
        )
        
        # Normalize to -1 to 1 range
        result.iloc[i] = np.tanh(composite_factor * 10)  # Scale and tanh for bounded output
    
    return result
