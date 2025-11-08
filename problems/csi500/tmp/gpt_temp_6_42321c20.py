import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Momentum-Volume Convergence Divergence Factor
    """
    df = data.copy()
    
    # Price Momentum Analysis
    # Short-term momentum (5-day)
    df['price_return_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_direction_5d'] = np.sign(df['price_return_5d'])
    
    # Medium-term momentum (20-day)
    df['price_return_20d'] = df['close'] / df['close'].shift(20) - 1
    df['price_direction_20d'] = np.sign(df['price_return_20d'])
    
    # Momentum consistency assessment
    df['momentum_aligned'] = (df['price_direction_5d'] == df['price_direction_20d']).astype(int)
    
    # Volume Momentum Analysis
    # Short-term volume dynamics
    df['volume_ratio_5d'] = df['volume'] / df['volume'].shift(5)
    
    # Calculate volume trend using linear regression slope (5-day)
    def volume_trend_5d(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(5)
        y = volume_series.values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    df['volume_trend_5d'] = df['volume'].rolling(window=5, min_periods=5).apply(volume_trend_5d, raw=False)
    df['volume_direction_5d'] = np.sign(df['volume_trend_5d'])
    
    # Medium-term volume dynamics
    df['volume_ratio_20d'] = df['volume'] / df['volume'].shift(20)
    
    # Calculate volume trend using linear regression slope (20-day)
    def volume_trend_20d(volume_series):
        if len(volume_series) < 20:
            return np.nan
        x = np.arange(20)
        y = volume_series.values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    df['volume_trend_20d'] = df['volume'].rolling(window=20, min_periods=20).apply(volume_trend_20d, raw=False)
    df['volume_direction_20d'] = np.sign(df['volume_trend_20d'])
    
    # Volume consistency assessment
    df['volume_aligned'] = (df['volume_direction_5d'] == df['volume_direction_20d']).astype(int)
    
    # Convergence-Divergence Detection Engine
    # Price-Volume alignment analysis
    df['short_term_alignment'] = df['price_direction_5d'] * df['volume_direction_5d']
    df['medium_term_alignment'] = df['price_direction_20d'] * df['volume_direction_20d']
    
    # Convergence strength calculations
    df['short_term_convergence'] = df['price_return_5d'] * df['volume_ratio_5d']
    df['medium_term_convergence'] = df['price_return_20d'] * df['volume_ratio_20d']
    
    # Timeframe consistency
    df['alignment_consistency'] = (df['short_term_alignment'] == df['medium_term_alignment']).astype(int)
    
    # Factor Synthesis
    # Primary convergence signal
    df['base_convergence'] = df['short_term_convergence'] + df['medium_term_convergence']
    
    # Directional weighting
    df['directional_weight'] = df['alignment_consistency'] * df['momentum_aligned'] * df['volume_aligned']
    
    # Volume confirmation adjustment
    volume_trend_magnitude = (np.abs(df['volume_trend_5d']) + np.abs(df['volume_trend_20d'])) / 2
    df['volume_confirmation'] = volume_trend_magnitude / (volume_trend_magnitude.rolling(window=20, min_periods=1).mean() + 1e-8)
    
    # Divergence adjustment
    df['signal_divergence'] = np.abs(df['short_term_alignment'] - df['medium_term_alignment']) / 2
    df['divergence_penalty'] = 1 - df['signal_divergence']
    
    # Final factor generation
    df['momentum_magnitude'] = (np.abs(df['price_return_5d']) + np.abs(df['price_return_20d'])) / 2
    
    # Combine all components
    factor = (df['base_convergence'] * 
              df['directional_weight'] * 
              df['volume_confirmation'] * 
              df['divergence_penalty'] * 
              df['momentum_magnitude'])
    
    return factor
