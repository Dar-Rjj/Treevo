import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Directional Momentum Strength
    df['bullish_momentum'] = np.where(df['returns'] > 0, df['returns'], 0).rolling(window=5, min_periods=3).sum()
    df['bearish_momentum'] = np.where(df['returns'] < 0, -df['returns'], 0).rolling(window=5, min_periods=3).sum()
    df['momentum_asymmetry'] = (df['bullish_momentum'] - df['bearish_momentum']) / (df['bullish_momentum'] + df['bearish_momentum'] + 1e-8)
    
    # Momentum persistence
    df['up_days'] = (df['returns'] > 0).astype(int)
    df['down_days'] = (df['returns'] < 0).astype(int)
    
    def consecutive_count(series):
        count = series.copy()
        for i in range(1, len(series)):
            if series.iloc[i] == 1:
                count.iloc[i] = count.iloc[i-1] + 1
            else:
                count.iloc[i] = 0
        return count
    
    df['consecutive_up'] = consecutive_count(df['up_days'])
    df['consecutive_down'] = consecutive_count(df['down_days'])
    df['momentum_persistence'] = df['consecutive_up'] - df['consecutive_down']
    
    # Price Structure Asymmetry
    df['upper_efficiency'] = (df['high'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['lower_efficiency'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['structure_bias'] = df['upper_efficiency'] - df['lower_efficiency']
    
    # Gap-direction alignment
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_alignment'] = np.sign(df['gap']) * df['structure_bias']
    
    # Volume Flow Asymmetry
    df['up_volume'] = np.where(df['returns'] > 0, df['volume'], 0)
    df['down_volume'] = np.where(df['returns'] < 0, df['volume'], 0)
    
    df['up_volume_conc'] = df['up_volume'].rolling(window=5, min_periods=3).sum() / df['volume'].rolling(window=5, min_periods=3).sum()
    df['down_volume_conc'] = df['down_volume'].rolling(window=5, min_periods=3).sum() / df['volume'].rolling(window=5, min_periods=3).sum()
    df['volume_flow_bias'] = df['up_volume_conc'] - df['down_volume_conc']
    
    # Volume persistence
    df['up_volume_days'] = (df['up_volume'] > 0).astype(int)
    df['down_volume_days'] = (df['down_volume'] > 0).astype(int)
    df['consecutive_up_volume'] = consecutive_count(df['up_volume_days'])
    df['consecutive_down_volume'] = consecutive_count(df['down_volume_days'])
    df['volume_persistence'] = df['consecutive_up_volume'] - df['consecutive_down_volume']
    
    # Multi-Timeframe Structure
    df['structure_1d'] = df['structure_bias']
    df['structure_3d'] = df['structure_bias'].rolling(window=3, min_periods=2).mean()
    df['structure_5d'] = df['structure_bias'].rolling(window=5, min_periods=3).mean()
    df['timeframe_convergence'] = (df['structure_1d'] * df['structure_3d'] * df['structure_5d'])
    
    # Regime Detection
    df['volatility'] = df['returns'].rolling(window=10, min_periods=5).std()
    df['trend_strength'] = df['returns'].rolling(window=10, min_periods=5).mean() / (df['volatility'] + 1e-8)
    
    # Regime-dependent asymmetry
    high_vol_mask = df['volatility'] > df['volatility'].rolling(window=20, min_periods=10).quantile(0.7)
    low_vol_mask = df['volatility'] < df['volatility'].rolling(window=20, min_periods=10).quantile(0.3)
    trending_mask = abs(df['trend_strength']) > 0.5
    mean_reverting_mask = abs(df['trend_strength']) < 0.2
    
    df['high_vol_asymmetry'] = np.where(high_vol_mask, df['momentum_asymmetry'] * df['structure_bias'], 0)
    df['low_vol_asymmetry'] = np.where(low_vol_mask, df['volume_flow_bias'] * df['structure_bias'], 0)
    df['trending_asymmetry'] = np.where(trending_mask, df['momentum_asymmetry'] * df['volume_flow_bias'], 0)
    df['mean_reverting_asymmetry'] = np.where(mean_reverting_mask, -df['structure_bias'] * df['volume_flow_bias'], 0)
    
    # Divergence Detection
    df['momentum_structure_div'] = df['momentum_asymmetry'] - df['structure_bias']
    df['volume_structure_div'] = df['volume_flow_bias'] - df['structure_bias']
    df['multi_timeframe_div'] = abs(df['structure_1d'] - df['structure_5d'])
    
    # Signal Synthesis
    df['asymmetry_convergence'] = (
        df['momentum_asymmetry'] * df['structure_bias'] * df['volume_flow_bias'] * 
        df['timeframe_convergence']
    )
    
    df['divergence_reversal'] = (
        -df['momentum_structure_div'] * df['volume_structure_div'] * 
        df['multi_timeframe_div']
    )
    
    # Regime-adaptive weights
    df['regime_weight'] = (
        df['high_vol_asymmetry'] + df['low_vol_asymmetry'] + 
        df['trending_asymmetry'] + df['mean_reverting_asymmetry']
    )
    
    # Composite Asymmetry Factor
    composite_factor = (
        0.3 * df['asymmetry_convergence'] +
        0.25 * df['divergence_reversal'] +
        0.2 * df['regime_weight'] +
        0.15 * df['momentum_persistence'] +
        0.1 * df['volume_persistence']
    )
    
    # Normalize and clean
    composite_factor = (composite_factor - composite_factor.rolling(window=20, min_periods=10).mean()) / (composite_factor.rolling(window=20, min_periods=10).std() + 1e-8)
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return composite_factor
