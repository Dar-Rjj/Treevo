import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate adaptive alpha factor combining multi-timeframe price-volume divergence,
    volatility regime detection, volume anomalies, and intraday momentum structure.
    """
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Price-Volume Divergence
        # Short-Term (3-Day)
        if i >= 3:
            price_return_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1
            volume_change_3d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3] - 1
            divergence_3d = price_return_3d - volume_change_3d
        else:
            divergence_3d = 0
        
        # Medium-Term (10-Day)
        if i >= 10:
            price_return_10d = current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1
            volume_change_10d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-10] - 1
            divergence_10d = price_return_10d - volume_change_10d
        else:
            divergence_10d = 0
        
        # Volatility Regime Detection
        if i >= 10:
            short_term_vol = returns.iloc[i-2:i+1].std() if i >= 2 else 0
            medium_term_vol = returns.iloc[i-9:i+1].std()
            
            if medium_term_vol > 0:
                volatility_ratio = short_term_vol / medium_term_vol
                
                # Regime Classification
                if volatility_ratio > 1.8:
                    # High Volatility: 80% Short-Term + 20% Medium-Term
                    regime_weighted_divergence = 0.8 * divergence_3d + 0.2 * divergence_10d
                elif volatility_ratio >= 0.8:
                    # Normal Volatility: 50% Short-Term + 50% Medium-Term
                    regime_weighted_divergence = 0.5 * divergence_3d + 0.5 * divergence_10d
                else:
                    # Low Volatility: 20% Short-Term + 80% Medium-Term
                    regime_weighted_divergence = 0.2 * divergence_3d + 0.8 * divergence_10d
            else:
                regime_weighted_divergence = 0.5 * divergence_3d + 0.5 * divergence_10d
        else:
            regime_weighted_divergence = divergence_3d if i >= 3 else 0
        
        # Volume Anomaly Detection
        if i >= 15:
            volume_window = current_data['volume'].iloc[i-14:i+1]
            volume_median = volume_window.median()
            volume_mad = (volume_window - volume_median).abs().median()
            
            current_volume = current_data['volume'].iloc[i]
            
            if volume_mad > 0:
                if current_volume > volume_median + 3 * volume_mad:
                    anomaly_score = 2.0
                elif current_volume > volume_median + 2 * volume_mad:
                    anomaly_score = 1.5
                else:
                    anomaly_score = 1.0
            else:
                anomaly_score = 1.0
        else:
            anomaly_score = 1.0
        
        # Volume-Enhanced Signal
        volume_enhanced_signal = regime_weighted_divergence * anomaly_score
        
        # Intraday Momentum Structure
        if i >= 1:
            gap_size = abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1]
            gap_direction = np.sign(current_data['open'].iloc[i] - current_data['close'].iloc[i-1])
            
            high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            if high_low_range > 0:
                range_efficiency = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range
            else:
                range_efficiency = 0
            
            momentum_quality = gap_direction * range_efficiency
            
            # Strong Signal Detection
            if gap_size > 0.015 and abs(range_efficiency) > 0.3:
                # Final Alpha with Momentum Quality adjustment for Strong Signals
                final_alpha = volume_enhanced_signal * (1 + momentum_quality)
            else:
                # Use Volume-Enhanced Signal otherwise
                final_alpha = volume_enhanced_signal
        else:
            final_alpha = volume_enhanced_signal
        
        result.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
