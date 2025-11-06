import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['micro_fractal_momentum'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) - 
                                   (df['close'].shift(1) - df['open'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8))
    
    # Meso Fractal Momentum
    df['high_3d'] = df['high'].rolling(window=4, min_periods=4).max()
    df['low_3d'] = df['low'].rolling(window=4, min_periods=4).min()
    df['high_6d'] = df['high'].shift(3).rolling(window=4, min_periods=4).max()
    df['low_6d'] = df['low'].shift(3).rolling(window=4, min_periods=4).min()
    
    df['meso_fractal_momentum'] = ((df['close'] - df['close'].shift(3)) / (df['high_3d'] - df['low_3d'] + 1e-8) - 
                                  (df['close'].shift(3) - df['close'].shift(6)) / (df['high_6d'] - df['low_6d'] + 1e-8))
    
    # Macro Fractal Momentum
    df['high_8d'] = df['high'].rolling(window=9, min_periods=9).max()
    df['low_8d'] = df['low'].rolling(window=9, min_periods=9).min()
    df['high_16d'] = df['high'].shift(8).rolling(window=9, min_periods=9).max()
    df['low_16d'] = df['low'].shift(8).rolling(window=9, min_periods=9).min()
    
    df['macro_fractal_momentum'] = ((df['close'] - df['close'].shift(8)) / (df['high_8d'] - df['low_8d'] + 1e-8) - 
                                   (df['close'].shift(8) - df['close'].shift(16)) / (df['high_16d'] - df['low_16d'] + 1e-8))
    
    # Fractal Momentum Cascade
    df['fractal_momentum_cascade'] = (df['micro_fractal_momentum'] * 
                                     df['meso_fractal_momentum'] * 
                                     df['macro_fractal_momentum'])
    
    # Volume Pressure Components
    df['buy_pressure'] = df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)
    df['sell_pressure'] = df['volume'] * (df['high'] - df['close']) / (df['high'] - df['low'] + 0.001)
    df['pressure_asymmetry'] = (df['buy_pressure'] - df['sell_pressure']) * np.sign(df['buy_pressure'] - df['sell_pressure'].shift(1))
    
    # Volume Momentum Fracture
    df['volume_momentum_fracture'] = ((df['close'] - df['open']) * df['volume'] / (df['high'] - df['low'] + 1e-8) - 
                                     (df['close'].shift(1) - df['open'].shift(1)) * df['volume'].shift(1) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8))
    
    # Volume-Fractal Alignment
    df['volume_fractal_alignment'] = (np.sign(df['volume_momentum_fracture']) * 
                                     np.sign(df['pressure_asymmetry']) * 
                                     np.power(df['volume'], 0.3))
    
    # Multi-Scale Fractal Efficiency
    df['micro_fractal_efficiency'] = np.abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8)
    
    df['high_4d'] = df['high'].rolling(window=5, min_periods=5).max()
    df['low_4d'] = df['low'].rolling(window=5, min_periods=5).min()
    df['meso_fractal_efficiency'] = np.abs(df['close'] - df['close'].shift(4)) / (df['high_4d'] - df['low_4d'] + 1e-8)
    
    df['high_10d'] = df['high'].rolling(window=11, min_periods=11).max()
    df['low_10d'] = df['low'].rolling(window=11, min_periods=11).min()
    df['macro_fractal_efficiency'] = np.abs(df['close'] - df['close'].shift(10)) / (df['high_10d'] - df['low_10d'] + 1e-8)
    
    df['fractal_efficiency_cascade'] = (df['micro_fractal_efficiency'] * 
                                       df['meso_fractal_efficiency'] * 
                                       df['macro_fractal_efficiency'])
    
    # Price Efficiency Asymmetry
    df['opening_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
    df['closing_efficiency'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 0.001)
    df['efficiency_asymmetry'] = df['opening_efficiency'] - df['closing_efficiency']
    
    # Volatility-Weighted Fractal Efficiency
    df['true_range'] = np.maximum(df['high'] - df['low'], 
                                 np.maximum(np.abs(df['high'] - df['close'].shift(1)), 
                                           np.abs(df['low'] - df['close'].shift(1))))
    df['atr_3d'] = df['true_range'].rolling(window=3, min_periods=3).mean()
    df['volatility_adjusted_fractal'] = df['fractal_efficiency_cascade'] / (df['atr_3d'] + 1e-8)
    
    # Asymmetric Volatility Transmission
    df['up_day_volatility'] = (df['high'] - df['open']) / (df['close'] - df['low'] + 0.001)
    df['down_day_volatility'] = (df['open'] - df['low']) / (df['high'] - df['close'] + 0.001)
    df['volatility_asymmetry_ratio'] = df['up_day_volatility'] - df['down_day_volatility']
    
    df['micro_volatility_expansion'] = ((df['high'] - df['low']) / (df['high'].shift(3) - df['low'].shift(3)) - 1) * (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 0.001)
    
    df['meso_volatility_momentum'] = ((df['high'] - df['low']) - (df['high'].shift(2) - df['low'].shift(2))) * (df['close'] - df['close'].shift(2)) / (df['high'] - df['low'] + 0.001)
    
    df['fractal_transmission_score'] = (np.sign(df['micro_volatility_expansion']) * 
                                       np.sign(df['meso_volatility_momentum']) * 
                                       np.power(df['high'] - df['low'], 0.5))
    
    # Regime-Adaptive Asymmetric Transmission
    df['daily_range_volatility'] = (df['high'] - df['low']) / df['close']
    df['volatility_momentum'] = df['daily_range_volatility'] - (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    
    df['high_volatility'] = df['daily_range_volatility'] > (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    df['low_volatility'] = df['daily_range_volatility'] < (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    
    # Volume Trend Components
    df['volume_trend_short'] = df['volume'] / df['volume'].shift(1)
    df['volume_trend_medium'] = df['volume'] / df['volume'].shift(3)
    df['volume_trend_long'] = df['volume'] / df['volume'].shift(8)
    
    # Volume Trend Consistency (simplified)
    def count_consistent_signs(series, window):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                window_data = series.iloc[i-window+1:i+1]
                short_signs = np.sign(window_data['volume_trend_short'])
                medium_signs = np.sign(window_data['volume_trend_medium'])
                long_signs = np.sign(window_data['volume_trend_long'])
                consistent = ((short_signs == medium_signs) & (medium_signs == long_signs)).sum()
                result.iloc[i] = consistent
        return result
    
    df['volume_trend_consistency'] = count_consistent_signs(df, 3)
    
    # High Volatility Transmission
    df['high_vol_transmission'] = (df['fractal_momentum_cascade'] * 
                                  df['fractal_transmission_score'] * 
                                  df['volume_fractal_alignment'] * 
                                  df['volatility_asymmetry_ratio'])
    
    # Low Volatility Transmission
    df['low_vol_transmission'] = (df['fractal_momentum_cascade'] * 
                                 df['fractal_efficiency_cascade'] * 
                                 df['volume_fractal_alignment'] * 
                                 df['efficiency_asymmetry'])
    
    # Transition Transmission
    df['transition_transmission'] = (df['fractal_momentum_cascade'] * 
                                    df['volatility_momentum'] * 
                                    df['volume_fractal_alignment'] * 
                                    df['volume_trend_consistency'])
    
    # Core Transmission
    df['core_transmission'] = np.where(df['high_volatility'], df['high_vol_transmission'],
                                      np.where(df['low_volatility'], df['low_vol_transmission'],
                                              df['transition_transmission']))
    
    # Volatility Enhanced
    df['volatility_enhanced'] = df['core_transmission'] / (df['atr_3d'] + 1e-8)
    
    # Asymmetric Fractal Persistence Analysis (simplified)
    df['micro_fractal_momentum_pos'] = (df['micro_fractal_momentum'] > 0).astype(int)
    df['micro_fractal_momentum_neg'] = (df['micro_fractal_momentum'] < 0).astype(int)
    
    df['short_term_persistence'] = (df['micro_fractal_momentum_pos'].rolling(window=3, min_periods=3).sum() - 
                                   df['micro_fractal_momentum_neg'].rolling(window=3, min_periods=3).sum())
    
    df['medium_term_persistence'] = (df['micro_fractal_momentum_pos'].rolling(window=6, min_periods=6).sum() - 
                                    df['micro_fractal_momentum_neg'].rolling(window=6, min_periods=6).sum())
    
    df['fracture_persistence_ratio'] = df['short_term_persistence'] / (df['medium_term_persistence'] + 1e-8)
    
    # Fractal Quality Score (simplified)
    df['price_fractal_consistency'] = ((df['high'] - df['low']) / (df['high'].rolling(window=3, min_periods=3).max() - df['low'].rolling(window=3, min_periods=3).min() + 0.001) * 
                                      (df['high'].rolling(window=3, min_periods=3).max() - df['low'].rolling(window=3, min_periods=3).min()) / 
                                      (df['high'].rolling(window=6, min_periods=6).max() - df['low'].rolling(window=6, min_periods=6).min() + 0.001))
    
    df['volume_fractal_consistency'] = (df['volume'] / df['volume'].shift(2)) * (df['volume'] / df['volume'].shift(5)) * (df['volume'] / df['volume'].shift(8))
    
    # Count consecutive days with same sign (simplified)
    def count_consecutive_sign(series):
        result = pd.Series(index=series.index, dtype=float)
        current_count = 0
        current_sign = 0
        for i, val in enumerate(series):
            if np.isnan(val):
                result.iloc[i] = 0
                continue
            sign_val = np.sign(val)
            if sign_val == current_sign:
                current_count += 1
            else:
                current_count = 1
                current_sign = sign_val
            result.iloc[i] = current_count
        return result
    
    df['momentum_cascade_sign_count'] = count_consecutive_sign(df['fractal_momentum_cascade'])
    
    df['fractal_quality_score'] = (df['price_fractal_consistency'] * 
                                  df['volume_fractal_consistency'] * 
                                  df['momentum_cascade_sign_count'])
    
    # Quality Confirmed
    df['quality_confirmed'] = df['volatility_enhanced'] * df['fractal_quality_score']
    
    # Persistence Adjusted
    df['persistence_adjusted'] = df['quality_confirmed'] * df['fracture_persistence_ratio']
    
    # Efficiency Enhanced
    df['efficiency_enhanced'] = df['persistence_adjusted'] * df['volatility_adjusted_fractal']
    
    # Asymmetric Strength Multiplier
    df['asymmetric_strength_multiplier'] = (np.abs(df['volume_fractal_alignment']) * 
                                           np.abs(df['fractal_transmission_score']) * 
                                           np.abs(df['pressure_asymmetry']))
    
    # Final Alpha
    alpha = (df['efficiency_enhanced'] * 
            (1 + df['asymmetric_strength_multiplier']) * 
            np.sign(df['fractal_momentum_cascade']) * 
            df['efficiency_asymmetry'])
    
    return alpha
