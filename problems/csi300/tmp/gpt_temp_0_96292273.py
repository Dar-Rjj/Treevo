import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Dimensional Momentum Acceleration with Confidence Alignment alpha factor
    """
    # Core Momentum Acceleration Components
    df['momentum_1d'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['momentum_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['momentum_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Acceleration Gradient Analysis
    df['accel_gradient_1'] = df['momentum_3d'] - df['momentum_1d']
    df['accel_gradient_2'] = df['momentum_5d'] - df['momentum_3d']
    
    # Volume-Price Alignment Engine
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volume_accel'] = df['volume'] / df['volume'].shift(3)
    
    # Volume persistence (consecutive days volume > previous day)
    df['volume_increase'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    df['volume_persistence'] = df['volume_increase'].groupby(df.index).expanding().apply(
        lambda x: (x == 1).cumsum().iloc[-1] if len(x) > 0 else 0
    ).reset_index(level=0, drop=True)
    
    # Price-Volume Divergence Detection
    df['bullish_divergence'] = ((df['momentum_3d'] > 0) & (df['volume_ratio'] < 1)).astype(int)
    df['bearish_divergence'] = ((df['momentum_3d'] < 0) & (df['volume_ratio'] > 1)).astype(int)
    df['volume_confirmation'] = ((df['momentum_3d'] > 0) & (df['volume_ratio'] > 1)) | ((df['momentum_3d'] < 0) & (df['volume_ratio'] < 1))
    
    # Volume-Weighted Momentum
    df['volume_aligned_momentum'] = df['momentum_3d'] * df['volume']
    df['volume_confirmed_signal'] = df['momentum_3d'] * df['volume_ratio']
    
    # Divergence penalty
    df['divergence_penalty'] = 1.0
    df.loc[df['bullish_divergence'] == 1, 'divergence_penalty'] = 0.7
    df.loc[df['bearish_divergence'] == 1, 'divergence_penalty'] = 0.7
    
    # Volatility-Scaled Persistence Framework
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['closing_strength'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Directional consistency (consecutive days with same 3-day momentum sign)
    df['momentum_sign'] = np.sign(df['momentum_3d'])
    df['momentum_sign_change'] = (df['momentum_sign'] != df['momentum_sign'].shift(1)).astype(int)
    df['directional_consistency'] = df.groupby(df.index)['momentum_sign_change'].expanding().apply(
        lambda x: (x == 0).cumsum().iloc[-1] if len(x) > 0 else 0
    ).reset_index(level=0, drop=True)
    
    # Acceleration persistence (consecutive days with positive acceleration gradient)
    df['accel_positive'] = (df['accel_gradient_1'] > 0).astype(int)
    df['accel_persistence'] = df['accel_positive'].groupby(df.index).expanding().apply(
        lambda x: (x == 1).cumsum().iloc[-1] if len(x) > 0 else 0
    ).reset_index(level=0, drop=True)
    
    # Adaptive Volatility Scaling
    df['volatility_adjusted_momentum'] = df['momentum_3d'] / df['daily_range'].replace(0, np.nan)
    df['persistence_score'] = (df['directional_consistency'] + df['accel_persistence'] + df['volume_persistence']) / 3
    
    # Confidence Alignment Integration
    # Price Confidence
    df['price_confidence'] = 0.3  # low by default
    df.loc[((df['momentum_3d'] > 0) & (df['closing_strength'] > 0.6)) | 
           ((df['momentum_3d'] < 0) & (df['closing_strength'] < 0.4)), 'price_confidence'] = 0.7  # medium
    df.loc[((df['momentum_3d'] > 0) & (df['closing_strength'] > 0.8)) | 
           ((df['momentum_3d'] < 0) & (df['closing_strength'] < 0.2)), 'price_confidence'] = 1.0  # high
    
    # Volume Confidence
    df['volume_confidence'] = 0.3  # low by default
    df.loc[df['volume_confirmation'], 'volume_confidence'] = 0.7  # medium
    df.loc[(df['volume_confirmation'] & (df['volume_ratio'] > 1.5)), 'volume_confidence'] = 1.0  # high
    
    # Persistence Confidence
    df['persistence_confidence'] = 0.3  # low by default
    df.loc[df['directional_consistency'] >= 2, 'persistence_confidence'] = 0.7  # medium
    df.loc[df['directional_consistency'] >= 4, 'persistence_confidence'] = 1.0  # high
    
    # Volatility Confidence (stable daily range)
    df['range_std_3d'] = df['daily_range'].rolling(window=3, min_periods=1).std()
    df['volatility_confidence'] = 0.3  # low by default
    df.loc[df['range_std_3d'] < df['daily_range'].quantile(0.6), 'volatility_confidence'] = 0.7  # medium
    df.loc[df['range_std_3d'] < df['daily_range'].quantile(0.3), 'volatility_confidence'] = 1.0  # high
    
    # Combined confidence product
    df['combined_confidence'] = (df['price_confidence'] * df['volume_confidence'] * 
                                df['persistence_confidence'] * df['volatility_confidence'])
    
    # Final Alpha Factor Construction
    # Core factor components
    df['core_factor'] = df['volatility_adjusted_momentum'] * (1 + df['accel_gradient_1'])
    df['volume_aligned_factor'] = df['core_factor'] * df['divergence_penalty']
    df['persistence_boosted_factor'] = df['volume_aligned_factor'] * (1 + df['persistence_score'] * 0.1)
    
    # Confidence filter application
    df['confidence_filtered'] = df['persistence_boosted_factor'] * df['combined_confidence']
    df.loc[df['combined_confidence'] < 0.5, 'confidence_filtered'] = 0
    df['range_efficiency_scaled'] = df['confidence_filtered'] * df['range_efficiency']
    
    # Multi-Timeframe Integration
    df['weighted_combination'] = (0.6 * df['momentum_3d'] + 
                                 0.3 * df['momentum_5d'] + 
                                 0.1 * df['momentum_1d'])
    
    # Final alpha output
    df['alpha_factor'] = df['weighted_combination'] * df['combined_confidence'] * df['range_efficiency_scaled']
    
    # Clean up and return
    alpha_series = df['alpha_factor'].copy()
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan)
    
    return alpha_series
