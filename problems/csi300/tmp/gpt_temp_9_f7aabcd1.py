import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close', 'amount', 'volume']
    if not all(col in df.columns for col in required_cols):
        return result
    
    # Create a copy to avoid modifying original
    data = df[required_cols].copy()
    
    for i in range(len(data)):
        if i < 6:  # Need at least 6 days for calculations
            result.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        prev_data = data.iloc[:i]  # Only past data
        
        # Liquidity Regime Classification
        # Volume-Price Shock Detection
        if len(prev_data) >= 5:
            vol_mean_5 = prev_data['volume'].iloc[-5:].mean()
            volume_spike = current['volume'] / vol_mean_5 if vol_mean_5 > 0 else 1
            
            prev_high_low_range = prev_data['high'].iloc[-1] - prev_data['low'].iloc[-1]
            curr_high_low_range = current['high'] - current['low']
            price_range_expansion = curr_high_low_range / prev_high_low_range if prev_high_low_range > 0 else 1
            
            liquidity_shock = volume_spike * price_range_expansion
            
            # Amount Efficiency Analysis
            amount_mean_5 = prev_data['amount'].iloc[-5:].mean()
            amount_intensity = current['amount'] / amount_mean_5 if amount_mean_5 > 0 else 1
            efficiency_ratio = amount_intensity / curr_high_low_range if curr_high_low_range > 0 else 1
            
            # Regime Classification
            if liquidity_shock > 2:
                regime_class = "High Stress"
            elif efficiency_ratio < 0.8:
                regime_class = "Low Efficiency"
            else:
                regime_class = "Normal"
        else:
            liquidity_shock = 1
            efficiency_ratio = 1
            regime_class = "Normal"
        
        # Multi-Timeframe Reversal Signals
        # Short-Term Reversal (2-day)
        if len(prev_data) >= 1:
            price_return = (current['close'] - prev_data['close'].iloc[-1]) / prev_data['close'].iloc[-1]
            vol_mean_2 = prev_data['volume'].iloc[-2:].mean() if len(prev_data) >= 2 else prev_data['volume'].iloc[-1]
            volume_confirmation = current['volume'] / vol_mean_2 if vol_mean_2 > 0 else 1
            short_reversal = -price_return * volume_confirmation
        else:
            short_reversal = 0
        
        # Medium-Term Reversal (6-day)
        if len(prev_data) >= 5:
            trend_return = (current['close'] - prev_data['close'].iloc[-5]) / prev_data['close'].iloc[-5]
            vol_mean_6 = prev_data['volume'].iloc[-6:].mean() if len(prev_data) >= 6 else prev_data['volume'].iloc[-5:].mean()
            volume_trend = current['volume'] / vol_mean_6 if vol_mean_6 > 0 else 1
            medium_reversal = -trend_return * volume_trend
        else:
            medium_reversal = 0
        
        reversal_convergence = short_reversal * medium_reversal
        
        # Volatility-Weighted Adjustment
        # True Range Calculation
        range1 = current['high'] - current['low']
        range2 = abs(current['high'] - prev_data['close'].iloc[-1]) if len(prev_data) >= 1 else range1
        range3 = abs(current['low'] - prev_data['close'].iloc[-1]) if len(prev_data) >= 1 else range1
        true_range = max(range1, range2, range3)
        
        # Volatility Persistence
        recent_volatility = true_range / current['close'] if current['close'] > 0 else 0
        
        if len(prev_data) >= 5:
            hist_true_range_mean = prev_data.apply(
                lambda x: max(x['high'] - x['low'], 
                            abs(x['high'] - prev_data['close'].shift(1).loc[x.name]) if x.name > prev_data.index[0] else x['high'] - x['low'],
                            abs(x['low'] - prev_data['close'].shift(1).loc[x.name]) if x.name > prev_data.index[0] else x['high'] - x['low']), 
                axis=1
            ).iloc[-5:].mean()
            hist_close_mean = prev_data['close'].iloc[-5:].mean()
            historical_volatility = hist_true_range_mean / hist_close_mean if hist_close_mean > 0 else recent_volatility
        else:
            historical_volatility = recent_volatility
        
        volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
        volatility_weight = 1 / volatility_ratio if volatility_ratio > 0 else 1
        
        # Order Flow Pressure Integration
        # Tick-Based Flow Direction
        if len(prev_data) >= 1:
            up_tick_volume = current['volume'] if current['close'] > prev_data['close'].iloc[-1] else 0
            down_tick_volume = current['volume'] if current['close'] < prev_data['close'].iloc[-1] else 0
            flow_imbalance = (up_tick_volume - down_tick_volume) / current['volume'] if current['volume'] > 0 else 0
        else:
            flow_imbalance = 0
        
        # Flow Persistence Pattern
        if len(prev_data) >= 3:
            consecutive_flow_days = 1
            for j in range(1, min(4, len(prev_data))):
                if i - j < 0:
                    break
                prev_flow = (prev_data['volume'].iloc[-j] if prev_data['close'].iloc[-j] > prev_data['close'].iloc[-j-1] else -prev_data['volume'].iloc[-j]) / prev_data['volume'].iloc[-j] if prev_data['volume'].iloc[-j] > 0 else 0
                curr_flow_sign = 1 if flow_imbalance > 0 else (-1 if flow_imbalance < 0 else 0)
                prev_flow_sign = 1 if prev_flow > 0 else (-1 if prev_flow < 0 else 0)
                if curr_flow_sign == prev_flow_sign and curr_flow_sign != 0:
                    consecutive_flow_days += 1
                else:
                    break
            
            prev_flow_imb = (prev_data['volume'].iloc[-1] if prev_data['close'].iloc[-1] > prev_data['close'].iloc[-2] else -prev_data['volume'].iloc[-1]) / prev_data['volume'].iloc[-1] if prev_data['volume'].iloc[-1] > 0 else 0
            flow_magnitude_change = abs(flow_imbalance / prev_flow_imb) if prev_flow_imb != 0 else 1
            persistence_strength = consecutive_flow_days * flow_magnitude_change
        else:
            persistence_strength = 1
        
        flow_pressure = flow_imbalance * persistence_strength
        
        # Adaptive Factor Combination
        # Regime-Weighted Components
        high_stress_multiplier = 2 if regime_class == "High Stress" else 1
        low_efficiency_multiplier = 1.5 if regime_class == "Low Efficiency" else 1
        regime_weight = high_stress_multiplier * low_efficiency_multiplier
        
        # Final calculations
        volatility_adjusted_reversal = reversal_convergence * volatility_weight
        flow_enhanced_signal = volatility_adjusted_reversal * flow_pressure
        efficiency_weighted_factor = flow_enhanced_signal * efficiency_ratio
        final_alpha = efficiency_weighted_factor * regime_weight * liquidity_shock
        
        result.iloc[i] = final_alpha
    
    return result
