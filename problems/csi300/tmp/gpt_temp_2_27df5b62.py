import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Asymmetry Framework
    df['directional_strength_1d'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # 3-day cumulative bias
    df['cumulative_bias_3d'] = df['directional_strength_1d'].rolling(window=3, min_periods=1).sum()
    
    # 5-day directional persistence
    df['sign_change'] = np.sign(df['close'] - df['open'])
    df['directional_persistence_5d'] = df['sign_change'].rolling(window=5, min_periods=1).apply(
        lambda x: np.mean(x == x[-1]) if len(x) == 5 else np.nan, raw=True
    )
    
    # Asymmetry acceleration patterns
    df['directional_momentum'] = df['cumulative_bias_3d'] - df['directional_strength_1d']
    df['asymmetry_volatility'] = df['directional_strength_1d'].rolling(window=5, min_periods=1).std()
    df['asymmetry_reversal'] = (np.sign(df['directional_strength_1d']) != np.sign(df['cumulative_bias_3d'])).astype(float)
    
    # Price-level microstructure
    df['open_close_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['high_low_range_expansion'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1)).replace(0, np.nan)
    df['gap_absorption'] = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume Microstructure Analysis
    df['volume_intensity'] = df['volume'] / df['volume'].shift(1).replace(0, np.nan)
    df['volume_momentum'] = df['volume'] / df['volume'].shift(3).replace(0, np.nan)
    df['volume_stability'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Volume-price microstructure relationship
    df['volume_weighted_directional_strength'] = df['directional_strength_1d'] * df['volume_intensity']
    
    # Volume-direction correlation (5-day)
    def volume_direction_correlation(x):
        if len(x) < 5:
            return np.nan
        volumes = df.loc[x.index, 'volume'].values
        strengths = df.loc[x.index, 'directional_strength_1d'].values
        return np.corrcoef(volumes, strengths)[0, 1] if not np.isnan(strengths).any() else np.nan
    
    df['volume_direction_correlation'] = df['directional_strength_1d'].rolling(window=5, min_periods=5).apply(
        volume_direction_correlation, raw=False
    )
    
    df['volume_directional_momentum'] = df['volume_momentum'] * df['directional_momentum']
    
    # Dynamic Microstructure Regime Detection
    df['trend_microstructure'] = ((df['close'] - df['close'].shift(5)) / 
                                 (df['high'].shift(5) - df['low'].shift(5)).replace(0, np.nan) > 0.1).astype(float)
    
    df['range_microstructure'] = ((df['high'] - df['low']) < 
                                 (df['high'] - df['low']).rolling(window=10, min_periods=1).mean()).astype(float)
    
    df['transition_microstructure'] = (np.abs(df['close'] - df['close'].rolling(window=10, min_periods=1).mean()) / 
                                      df['close'].rolling(window=10, min_periods=1).mean().replace(0, np.nan) > 0.02).astype(float)
    
    # Microstructure volatility patterns
    df['range_expansion_intensity'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=10, min_periods=1).mean()
    
    def volatility_clustering(x):
        if len(x) < 5:
            return np.nan
        count = 0
        for i in range(len(x)):
            if i >= 9:
                window_mean = x[i-9:i+1].mean()
                if x[i] > window_mean:
                    count += 1
        return count / len(x)
    
    df['volatility_clustering_microstructure'] = (df['high'] - df['low']).rolling(window=5, min_periods=5).apply(
        volatility_clustering, raw=True
    )
    
    df['volatility_persistence_microstructure'] = ((df['high'] - df['low']).rolling(window=5, min_periods=1).std() / 
                                                  (df['high'] - df['low']).rolling(window=10, min_periods=1).mean())
    
    # Directional microstructure regimes
    df['strong_directional_bias'] = (np.abs(df['directional_strength_1d']) > 0.7).astype(float)
    df['weak_directional_bias'] = (np.abs(df['directional_strength_1d']) < 0.3).astype(float)
    
    # Cross-Microstructure Signal Generation
    df['trend_directional_strength'] = df['directional_strength_1d'] * ((df['close'] - df['close'].shift(5)) / 
                                                                       (df['high'].shift(5) - df['low'].shift(5)).replace(0, np.nan))
    df['volume_confirmed_trend_microstructure'] = df['volume_intensity'] * df['trend_directional_strength']
    df['trend_acceleration_microstructure'] = df['directional_momentum'] * ((df['close'] - df['close'].shift(5)) / 
                                                                           (df['high'].shift(5) - df['low'].shift(5)).replace(0, np.nan))
    
    df['range_directional_efficiency'] = df['directional_strength_1d'] * df['range_expansion_intensity']
    df['volume_compression_microstructure'] = df['volume_stability'] * df['range_directional_efficiency']
    df['breakout_microstructure'] = df['volume_intensity'] * df['range_directional_efficiency']
    
    df['microstructure_change_momentum'] = df['directional_momentum'] * df['range_expansion_intensity']
    df['volume_microstructure_confirmation'] = df['volume_directional_momentum'] * df['microstructure_change_momentum']
    df['transition_directional_strength'] = df['directional_strength_1d'] * df['range_expansion_intensity']
    
    # Multi-Timeframe Microstructure Integration
    df['microstructure_strength_1d'] = df['directional_strength_1d'] * df['volume_intensity']
    df['microstructure_persistence_3d'] = df['cumulative_bias_3d'] * df['volume_momentum']
    df['microstructure_consistency_5d'] = df['asymmetry_volatility'] * df['volume_stability']
    
    df['microstructure_alignment'] = df['microstructure_strength_1d'] * df['microstructure_persistence_3d']
    df['microstructure_acceleration'] = df['microstructure_strength_1d'] - df['microstructure_persistence_3d']
    df['microstructure_divergence'] = np.abs(df['microstructure_strength_1d']) / np.abs(df['microstructure_persistence_3d']).replace(0, np.nan)
    
    df['trend_optimized_microstructure'] = df['trend_acceleration_microstructure'] * df['volume_confirmed_trend_microstructure']
    df['range_optimized_microstructure'] = df['range_directional_efficiency'] * df['volume_compression_microstructure']
    df['transition_optimized_microstructure'] = df['transition_directional_strength'] * df['volume_microstructure_confirmation']
    
    # Dynamic Microstructure Alpha Synthesis
    df['trend_microstructure_factor'] = df['trend_optimized_microstructure'] * df['microstructure_alignment']
    df['range_microstructure_factor'] = df['range_optimized_microstructure'] * df['microstructure_consistency_5d']
    df['transition_microstructure_factor'] = df['transition_optimized_microstructure'] * df['microstructure_acceleration']
    
    # Unified microstructure alpha output
    alpha = (df['trend_microstructure_factor'].fillna(0) * 0.4 + 
             df['range_microstructure_factor'].fillna(0) * 0.35 + 
             df['transition_microstructure_factor'].fillna(0) * 0.25)
    
    return alpha
