import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Efficiency Component Calculation
    df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_efficiency'] = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    df['combined_efficiency'] = df['price_efficiency'] * df['volume_efficiency']
    
    # Multi-Timeframe Momentum Analysis
    df['ultra_short_momentum'] = df['close'] / df['close'].shift(1) - 1
    df['short_term_momentum'] = df['close'] / df['close'].shift(3) - 1
    df['medium_term_momentum'] = df['close'] / df['close'].shift(8) - 1
    df['momentum_spectrum_spread'] = (df['ultra_short_momentum'] - df['medium_term_momentum']) / (df['medium_term_momentum'].abs().replace(0, np.nan))
    
    # Volume Dynamics Characterization
    df['volume_persistence'] = df['volume'].rolling(window=5).apply(lambda x: (x > x.shift(1)).sum(), raw=False)
    df['volume_clustering'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
    
    # Regime Identification System
    df['trend_strength'] = (df['close'] / df['close'].shift(10) - 1).abs()
    
    def trend_consistency_calc(series):
        if len(series) < 3:
            return np.nan
        signs = np.sign(series.diff())
        return (signs == signs.shift(1)).rolling(window=5).sum().iloc[-1]
    
    df['trend_consistency'] = df['close'].rolling(window=6).apply(trend_consistency_calc, raw=False)
    
    df['trend_regime'] = (df['trend_strength'] > 0.05) & (df['trend_consistency'] >= 3)
    
    df['range_stability'] = (df['high'] - df['low']).rolling(window=5).std() / (df['high'] - df['low']).rolling(window=5).mean()
    
    def price_oscillation_calc(df_window):
        if len(df_window) < 6:
            return np.nan
        count = 0
        for i in range(1, 6):
            daily_range = df_window['high'].iloc[i] - df_window['low'].iloc[i]
            if daily_range > 0:
                price_move = abs(df_window['close'].iloc[i] - df_window['close'].iloc[i-1])
                if price_move / daily_range < 0.3:
                    count += 1
        return count
    
    df['price_oscillation'] = df.rolling(window=6).apply(price_oscillation_calc, raw=False)
    df['range_bound_regime'] = (df['range_stability'] < 0.5) & (df['price_oscillation'] >= 3)
    
    # Efficiency-Momentum Integration
    df['efficiency_weighted_momentum'] = df['combined_efficiency'] * df['momentum_spectrum_spread']
    df['volume_confirmed_momentum'] = df['volume_momentum'] * df['ultra_short_momentum']
    df['persistence_adjusted_factor'] = df['efficiency_weighted_momentum'] * df['volume_persistence']
    
    # Regime-Specific Enhancement
    # Trend regime processing
    trend_factor = (df['efficiency_weighted_momentum'] * 
                   df['trend_strength'] * 
                   df['trend_consistency'] * 
                   df['volume_clustering'])
    
    # Range-bound regime processing
    range_factor = (df['persistence_adjusted_factor'] * 
                   df['price_efficiency'] * 
                   (2 - df['volume_clustering']) * 
                   df['price_oscillation'])
    
    # Cross-Regime Signal Refinement
    # Regime transition detection
    df['regime_change_count'] = (df['trend_regime'] != df['trend_regime'].shift(1)).rolling(window=3).sum()
    df['transition_penalty'] = 1 / (1 + df['regime_change_count'])
    
    # Signal consistency check
    def factor_persistence_calc(factor_series):
        if len(factor_series) < 4:
            return np.nan
        signs = np.sign(factor_series)
        return (signs == signs.shift(1)).tail(3).sum()
    
    df['factor_persistence'] = df['efficiency_weighted_momentum'].rolling(window=4).apply(factor_persistence_calc, raw=False)
    df['consistency_multiplier'] = 1 + df['factor_persistence'] / 3
    
    # Volume-price alignment
    df['alignment_score'] = np.sign(df['volume_momentum']) * np.sign(df['ultra_short_momentum'])
    df['alignment_multiplier'] = 1 + df['alignment_score'].abs() * 0.5
    
    # Final Alpha Synthesis
    regime_factor = np.where(df['trend_regime'], trend_factor, 
                           np.where(df['range_bound_regime'], range_factor, 
                                   df['efficiency_weighted_momentum']))
    
    final_alpha = (regime_factor * 
                  df['transition_penalty'] * 
                  df['consistency_multiplier'] * 
                  df['alignment_multiplier'])
    
    return final_alpha
