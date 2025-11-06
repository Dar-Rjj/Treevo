import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Range Efficiency factor
    Combines price momentum, range efficiency, volume confirmation, volatility regimes, and gap analysis
    """
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['prev_close'] = df['close'].shift(1)
    df['range'] = df['high'] - df['low']
    df['range_efficiency'] = (df['close'] - df['open']) / (df['range'] + 1e-8)
    
    # Average True Range for volatility
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['atr'] = pd.concat([df['tr1'], df['tr2'], df['tr3']], axis=1).max(axis=1)
    df['atr_21'] = df['atr'].rolling(window=21, min_periods=1).mean()
    
    # Timeframes for analysis
    timeframes = [3, 8, 21]
    
    # Calculate momentum and range efficiency for each timeframe
    for tf in timeframes:
        # Price momentum (rate of change)
        df[f'momentum_{tf}d'] = df['close'] / df['close'].shift(tf) - 1
        
        # Range efficiency rolling average
        df[f'range_eff_{tf}d'] = df['range_efficiency'].rolling(window=tf, min_periods=1).mean()
        
        # Volume rate of change
        df[f'volume_roc_{tf}d'] = df['volume'] / df['volume'].shift(tf) - 1
    
    # Calculate divergence between momentum and range efficiency
    for tf in timeframes:
        df[f'divergence_{tf}d'] = df[f'momentum_{tf}d'] / (abs(df[f'range_eff_{tf}d']) + 1e-8)
    
    # Volatility regime classification
    df['range_eff_21d_ma'] = df['range_efficiency'].rolling(window=21, min_periods=1).mean()
    df['range_eff_21d_std'] = df['range_efficiency'].rolling(window=21, min_periods=1).std()
    df['volatility_regime'] = (df['range_efficiency'] - df['range_eff_21d_ma']) / (df['range_eff_21d_std'] + 1e-8)
    
    # Gap analysis
    df['overnight_gap'] = (df['open'] - df['prev_close']) / (df['atr_21'] + 1e-8)
    df['gap_strength'] = abs(df['overnight_gap']) * np.sign(df['overnight_gap'])
    
    # Multi-timeframe alignment score
    divergence_signs = []
    for tf in timeframes:
        divergence_signs.append(np.sign(df[f'divergence_{tf}d']))
    
    # Calculate alignment (how many timeframes agree on direction)
    df['alignment_score'] = pd.DataFrame(divergence_signs).T.sum(axis=1) / len(timeframes)
    
    # Volume confirmation across timeframes
    volume_confirmation = []
    for tf in timeframes:
        vol_conf = df[f'volume_roc_{tf}d'] * np.sign(df[f'divergence_{tf}d'])
        volume_confirmation.append(vol_conf)
    
    df['volume_confirmation'] = pd.DataFrame(volume_confirmation).T.mean(axis=1)
    
    # Regime-adaptive weighting
    df['regime_weight'] = np.where(
        df['volatility_regime'] > 1.0, 1.5,  # High efficiency - enhance signals
        np.where(df['volatility_regime'] < -1.0, 0.5, 1.0)  # Low efficiency - reduce noise
    )
    
    # Combine all components with temporal smoothing
    for i in range(len(df)):
        if i < 21:  # Ensure enough data for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        
        # Weighted divergence across timeframes (more weight to shorter timeframes)
        weighted_divergence = (
            0.5 * current_data['divergence_3d'] +
            0.3 * current_data['divergence_8d'] + 
            0.2 * current_data['divergence_21d']
        )
        
        # Combine with alignment and volume confirmation
        combined_signal = (
            weighted_divergence * 
            current_data['alignment_score'] * 
            (1 + current_data['volume_confirmation'])
        )
        
        # Apply gap enhancement
        gap_enhancement = 1 + (0.2 * current_data['gap_strength'])
        
        # Apply regime-adaptive weighting
        regime_adjusted = combined_signal * current_data['regime_weight'] * gap_enhancement
        
        # Apply temporal smoothing (3-day rolling window)
        if i >= 23:
            recent_signals = result.iloc[i-3:i]
            smoothed_signal = 0.6 * regime_adjusted + 0.4 * recent_signals.mean()
        else:
            smoothed_signal = regime_adjusted
        
        result.iloc[i] = smoothed_signal
    
    # Final normalization
    result = (result - result.rolling(window=21, min_periods=1).mean()) / (result.rolling(window=21, min_periods=1).std() + 1e-8)
    
    return result
