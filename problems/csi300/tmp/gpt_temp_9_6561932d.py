import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Fractal Divergence with Liquidity Shock Detection
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate multi-scale momentum patterns
    df['momentum_micro'] = df['close'].pct_change(2)
    df['momentum_meso'] = df['close'].pct_change(8)
    df['momentum_macro'] = df['close'].pct_change(21)
    
    # Calculate momentum variance ratios for fractal dimension
    df['momentum_var_micro'] = df['momentum_micro'].rolling(window=10, min_periods=5).var()
    df['momentum_var_meso'] = df['momentum_meso'].rolling(window=10, min_periods=5).var()
    df['momentum_var_macro'] = df['momentum_macro'].rolling(window=10, min_periods=5).var()
    
    # Compute momentum fractal dimension
    df['fractal_dimension'] = np.log(df['momentum_var_macro'] / df['momentum_var_micro']) / np.log(21/2)
    df['fractal_dimension'] = df['fractal_dimension'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate momentum acceleration gradient
    df['momentum_accel'] = df['momentum_micro'].diff(3)
    df['momentum_curvature'] = df['momentum_accel'].diff(3)
    
    # Identify momentum regime shifts
    df['momentum_regime'] = 0
    bull_condition = (df['momentum_micro'] > 0) & (df['momentum_accel'] > 0)
    bear_condition = (df['momentum_micro'] < 0) & (df['momentum_accel'] < 0)
    df.loc[bull_condition, 'momentum_regime'] = 1  # Bullish
    df.loc[bear_condition, 'momentum_regime'] = -1  # Bearish
    
    # Compute Fractal Divergence Score
    df['phase_intensity'] = np.abs(df['momentum_accel']) * np.abs(df['momentum_curvature'])
    df['fractal_divergence'] = df['fractal_dimension'] * df['phase_intensity'] * df['momentum_regime']
    
    # Calculate Liquidity Shock Pressure
    # Volume fractal anomalies
    df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_std_20'] = df['volume'].rolling(window=20, min_periods=10).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma_20']) / df['volume_std_20']
    df['volume_zscore'] = df['volume_zscore'].replace([np.inf, -np.inf], np.nan)
    
    # Volume shock intensity
    df['volume_spike_magnitude'] = np.abs(df['volume_zscore'])
    df['volume_persistence'] = df['volume_spike_magnitude'].rolling(window=5, min_periods=3).mean()
    
    # Liquidity absorption efficiency
    df['price_impact'] = np.abs(df['close'].pct_change()) / (df['volume'] + 1e-8)
    df['liquidity_efficiency'] = 1 / (df['price_impact'].rolling(window=10, min_periods=5).mean() + 1e-8)
    
    # Liquidity pressure signal
    df['liquidity_pressure'] = (df['volume_spike_magnitude'] * df['liquidity_efficiency'] * 
                               np.abs(df['volume_zscore']))
    
    # Detect Fractal Breakdown Signals
    # Multi-scale pattern divergence
    df['micro_meso_gap'] = np.abs(df['momentum_micro'] - df['momentum_meso'])
    df['meso_macro_gap'] = np.abs(df['momentum_meso'] - df['momentum_macro'])
    df['pattern_divergence'] = (df['micro_meso_gap'] + df['meso_macro_gap']) / 2
    
    # Volume-price fractal correlation
    df['volume_momentum_corr'] = df['volume'].rolling(window=15, min_periods=8).corr(df['momentum_micro'])
    df['correlation_break'] = 1 - np.abs(df['volume_momentum_corr'])
    
    # Breakdown alert score
    df['breakdown_alert'] = (0.7 * df['pattern_divergence'] + 0.3 * df['correlation_break'])
    
    # Synthesize Final Alpha Factor
    df['raw_factor'] = df['fractal_divergence'] * df['liquidity_pressure'] * df['breakdown_alert']
    
    # Fractal-Adaptive Smoothing
    df['fractal_dim_bin'] = pd.cut(df['fractal_dimension'], 
                                  bins=[-np.inf, 0.3, 0.7, np.inf], 
                                  labels=['low', 'medium', 'high'])
    
    # Apply pattern-complexity adjusted smoothing
    for idx in df.index:
        if pd.isna(df.loc[idx, 'fractal_dim_bin']):
            result.loc[idx] = np.nan
            continue
            
        fractal_bin = df.loc[idx, 'fractal_dim_bin']
        
        if fractal_bin == 'high':
            # Use shorter window for high fractal dimension (complex patterns)
            window_data = df.loc[:idx, 'raw_factor'].tail(3)
            result.loc[idx] = window_data.mean() if len(window_data) >= 2 else np.nan
        elif fractal_bin == 'low':
            # Use longer window for low fractal dimension (simple patterns)
            window_data = df.loc[:idx, 'raw_factor'].tail(11)
            result.loc[idx] = window_data.mean() if len(window_data) >= 6 else np.nan
        else:  # medium
            # Use medium window for medium fractal dimension
            window_data = df.loc[:idx, 'raw_factor'].tail(6)
            result.loc[idx] = window_data.mean() if len(window_data) >= 3 else np.nan
    
    return result
