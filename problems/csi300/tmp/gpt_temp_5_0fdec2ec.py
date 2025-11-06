import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-Scale Price Reversal Detection
    # Short-term extrema (t-5 to t-2)
    data['short_high'] = data['high'].rolling(window=4, min_periods=2).max().shift(2)
    data['short_low'] = data['low'].rolling(window=4, min_periods=2).min().shift(2)
    
    # Medium-term extrema (t-10 to t-3)
    data['medium_high'] = data['high'].rolling(window=8, min_periods=4).max().shift(3)
    data['medium_low'] = data['low'].rolling(window=8, min_periods=4).min().shift(3)
    
    # Calculate reversal distances
    data['short_rev_up'] = (data['short_high'] - data['close']) / data['close']
    data['short_rev_down'] = (data['close'] - data['short_low']) / data['close']
    data['medium_rev_up'] = (data['medium_high'] - data['close']) / data['close']
    data['medium_rev_down'] = (data['close'] - data['medium_low']) / data['close']
    
    # Combined reversal strength
    data['short_reversal'] = np.where(data['short_rev_up'] > data['short_rev_down'], 
                                     -data['short_rev_up'], data['short_rev_down'])
    data['medium_reversal'] = np.where(data['medium_rev_up'] > data['medium_rev_down'], 
                                      -data['medium_rev_up'], data['medium_rev_down'])
    
    # Volatility Context
    # Short-term volatility measures
    data['short_atr'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    data['short_hl_range'] = (data['high'].rolling(window=5, min_periods=3).max() - 
                             data['low'].rolling(window=5, min_periods=3).min()) / data['close']
    
    # Medium-term volatility measures
    data['medium_atr'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    data['medium_hl_range'] = (data['high'].rolling(window=20, min_periods=10).max() - 
                              data['low'].rolling(window=20, min_periods=10).min()) / data['close']
    
    # Volatility-adjusted reversals
    data['short_rev_vol_adj'] = data['short_reversal'] / (data['short_atr'] / data['close'] + 1e-8)
    data['medium_rev_vol_adj'] = data['medium_reversal'] / (data['medium_atr'] / data['close'] + 1e-8)
    
    # Volume Confirmation
    # Volume momentum
    data['volume_5d_change'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean() - 1
    data['volume_20d_trend'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 10 else np.nan
    )
    
    # Volume-price relationship
    data['volume_confirmation'] = np.where(
        (data['short_reversal'] > 0) & (data['volume_5d_change'] > 0), 1,
        np.where((data['short_reversal'] < 0) & (data['volume_5d_change'] > 0), -1, 0)
    )
    
    # Volume divergence detection
    data['volume_divergence'] = np.where(
        (data['short_reversal'] > 0) & (data['volume_5d_change'] < -0.1), -0.5,
        np.where((data['short_reversal'] < 0) & (data['volume_5d_change'] < -0.1), 0.5, 0)
    )
    
    # Composite Alpha Factor
    # Weight short-term vs medium-term reversal
    short_weight = 0.6
    medium_weight = 0.4
    
    # Volatility context weighting
    vol_regime = data['medium_atr'] / data['medium_atr'].rolling(window=50, min_periods=25).mean()
    vol_weight = np.where(vol_regime > 1.2, 0.7, np.where(vol_regime < 0.8, 1.3, 1.0))
    
    # Combine reversal components
    base_reversal = (short_weight * data['short_rev_vol_adj'] + 
                    medium_weight * data['medium_rev_vol_adj'])
    
    # Apply volatility regime sensitivity
    volatility_adjusted = base_reversal * vol_weight
    
    # Integrate volume confirmation
    volume_strength = 1 + (0.3 * data['volume_confirmation'] + 0.2 * data['volume_divergence'])
    volume_strength = np.clip(volume_strength, 0.5, 1.5)
    
    # Final factor with volume adjustment
    final_factor = volatility_adjusted * volume_strength
    
    # Clean up intermediate columns
    result = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
