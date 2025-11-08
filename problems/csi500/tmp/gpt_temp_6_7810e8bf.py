import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum-Efficiency Divergence factor
    """
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead
    for i in range(len(data)):
        if i < 20:  # Need enough history for calculations
            result.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        
        # Momentum Efficiency Component
        # Intraday Momentum Efficiency
        high_low_range = current['high'] - current['low']
        if high_low_range > 0:
            intraday_efficiency = (current['close'] - current['open']) / high_low_range
        else:
            intraday_efficiency = 0
            
        # Momentum Persistence
        if i >= 5:
            momentum_3d = current['close'] / data.iloc[i-3]['close'] if data.iloc[i-3]['close'] > 0 else 1
            momentum_5d = current['close'] / data.iloc[i-5]['close'] if data.iloc[i-5]['close'] > 0 else 1
        else:
            momentum_3d = momentum_5d = 1
            
        # Momentum Acceleration
        if i >= 10:
            short_roc = current['close'] / data.iloc[i-5]['close'] - 1 if data.iloc[i-5]['close'] > 0 else 0
            medium_roc = current['close'] / data.iloc[i-10]['close'] - 1 if data.iloc[i-10]['close'] > 0 else 0
            momentum_acceleration = short_roc - medium_roc
        else:
            momentum_acceleration = 0
            
        # Volume-Price Divergence Component
        # Volume Spike & Trend
        vol_median = np.median([data.iloc[j]['volume'] for j in range(i-20, i)])
        volume_spike = current['volume'] / vol_median if vol_median > 0 else 1
        
        if i >= 5:
            volume_trend = current['volume'] / data.iloc[i-5]['volume'] if data.iloc[i-5]['volume'] > 0 else 1
        else:
            volume_trend = 1
            
        if i >= 10:
            vol_accel_5d = current['volume'] / data.iloc[i-5]['volume'] if data.iloc[i-5]['volume'] > 0 else 1
            vol_accel_10d = current['volume'] / data.iloc[i-10]['volume'] if data.iloc[i-10]['volume'] > 0 else 1
            volume_acceleration = vol_accel_5d - vol_accel_10d
        else:
            volume_acceleration = 0
            
        # Price-Volume Alignment
        price_volume_alignment = 1 if np.sign(momentum_acceleration) == np.sign(volume_acceleration) else -1
        divergence_strength = momentum_5d * volume_spike * price_volume_alignment
        
        # Volume Confirmation Logic
        if volume_trend > 1.2 and price_volume_alignment > 0:
            volume_confirmation = 1.5
        elif volume_trend < 0.8 and price_volume_alignment < 0:
            volume_confirmation = 0.5
        else:
            volume_confirmation = 1.0
            
        # Volatility Efficiency Component
        # Volume-Weighted Volatility Dynamics
        if current['volume'] > 0:
            vol_eff_ratio = (current['high'] - current['low']) / current['volume']
        else:
            vol_eff_ratio = 0
            
        if i >= 2:
            vol_eff_prev = (data.iloc[i-2]['high'] - data.iloc[i-2]['low']) / data.iloc[i-2]['volume'] if data.iloc[i-2]['volume'] > 0 else 0
            vol_vol_persistence = vol_eff_ratio - vol_eff_prev
        else:
            vol_vol_persistence = 0
            
        regime_intensity = current['volume'] * (current['high'] - current['low'])
        
        # Price Momentum Under Volume Stress
        if i >= 1 and current['volume'] > 0:
            vol_adj_momentum = (current['close'] - data.iloc[i-1]['close']) / current['volume']
        else:
            vol_adj_momentum = 0
            
        if i >= 2 and current['volume'] > 0 and data.iloc[i-2]['volume'] > 0:
            vol_adj_prev = (data.iloc[i-2]['close'] - data.iloc[i-3]['close']) / data.iloc[i-2]['volume'] if i >= 3 else 0
            mom_sustainability = vol_adj_momentum - vol_adj_prev
        else:
            mom_sustainability = 0
            
        stress_signal = mom_sustainability * regime_intensity
        
        # Volatility Regime Analysis
        intraday_vol_ratio = (current['high'] - current['low']) / current['close'] if current['close'] > 0 else 0
        
        # Calculate 10-day average intraday range
        avg_intraday_range = np.mean([(data.iloc[j]['high'] - data.iloc[j]['low']) / data.iloc[j]['close'] 
                                    for j in range(max(0, i-10), i) if data.iloc[j]['close'] > 0])
        
        # Classify Volatility Environment
        if avg_intraday_range > 0:
            if intraday_vol_ratio > 1.5 * avg_intraday_range:
                volatility_regime = 'high'
                vol_adjustment = 0.5
            elif intraday_vol_ratio < 0.5 * avg_intraday_range:
                volatility_regime = 'low'
                vol_adjustment = 1.25
            else:
                volatility_regime = 'normal'
                vol_adjustment = 1.0
        else:
            volatility_regime = 'normal'
            vol_adjustment = 1.0
        
        # Efficiency-Weighted Signal Generation
        # Momentum Efficiency Integration
        momentum_efficiency = momentum_5d * intraday_efficiency
        if abs(intraday_efficiency) > 0.3:
            momentum_efficiency *= 1.2
        
        # Volatility Efficiency Integration
        core_efficiency = vol_vol_persistence * stress_signal
        
        if i >= 3:
            range_prev = data.iloc[i-3]['high'] - data.iloc[i-3]['low']
            range_current = current['high'] - current['low']
            if range_prev > 0:
                range_adjustment = range_current / range_prev
            else:
                range_adjustment = 1
        else:
            range_adjustment = 1
            
        volatility_efficiency = core_efficiency * range_adjustment
        
        # Multi-Timeframe Signal Integration
        # Short-term Signal (1-3 days)
        short_term_signal = intraday_efficiency * momentum_3d * 0.6
        
        # Medium-term Signal (4-10 days)
        medium_term_signal = momentum_5d * volume_trend * 0.4
        
        # Signal Consistency Check
        signal_alignment = 1 if np.sign(short_term_signal) == np.sign(medium_term_signal) else 0.5
        
        # Adaptive Composite Factor Generation
        # Volatility-Adaptive Signal Combination
        if volatility_regime == 'high':
            composite_signal = medium_term_signal * 0.7 + short_term_signal * 0.3
        elif volatility_regime == 'low':
            composite_signal = short_term_signal * 0.8 + medium_term_signal * 0.2
        else:
            composite_signal = short_term_signal * 0.5 + medium_term_signal * 0.5
        
        # Efficiency-Confirmed Signal Generation
        if abs(intraday_efficiency) < 0.1:
            composite_signal *= 0.5
            
        if volume_trend < 0.7:
            composite_signal *= 0.8
            
        if momentum_5d < 0.95:
            composite_signal *= 0.9
            
        # Apply volatility adjustment
        composite_signal *= vol_adjustment
        composite_signal *= signal_alignment
        
        # Risk-Adjusted Final Factor
        # Calculate True Range
        true_range = current['high'] - current['low']
        if i >= 1:
            true_range = max(
                true_range,
                abs(current['high'] - data.iloc[i-1]['close']),
                abs(current['low'] - data.iloc[i-1]['close'])
            )
        
        # Signal-to-noise ratio
        if i >= 5:
            recent_volatility = np.std([data.iloc[j]['close'] for j in range(i-5, i)])
            if recent_volatility > 0:
                signal_to_noise = abs(composite_signal) / recent_volatility
                if signal_to_noise < 0.1:
                    composite_signal *= 0.5
                elif signal_to_noise > 0.5:
                    composite_signal *= 1.2
        
        # Final factor with risk adjustment
        if true_range > 0:
            final_factor = composite_signal / true_range
        else:
            final_factor = composite_signal
            
        result.iloc[i] = final_factor
    
    return result
