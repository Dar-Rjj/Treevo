import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Amplitude-Volume Divergence Factor
    Combines amplitude structure analysis with volume-amplitude divergence detection
    in a regime-adaptive framework for return prediction.
    """
    df = df.copy()
    
    # Calculate basic amplitude metrics
    df['amplitude'] = (df['high'] - df['low']) / df['open']
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
    df['amplitude_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Amplitude Structure Analysis
    # Short-term amplitude asymmetry (geometric mean from t-1 to t)
    df['short_term_amplitude'] = df['amplitude'].rolling(window=2).apply(
        lambda x: np.exp(np.mean(np.log(x + 1e-8))) if len(x) == 2 else np.nan
    )
    
    # Medium-term amplitude trend (geometric mean from t-5 to t)
    df['medium_term_amplitude'] = df['amplitude'].rolling(window=5).apply(
        lambda x: np.exp(np.mean(np.log(x + 1e-8))) if len(x) == 5 else np.nan
    )
    
    # Regime classification (short-term / medium-term amplitude ratio)
    df['amplitude_regime'] = df['short_term_amplitude'] / df['medium_term_amplitude']
    
    # Volume-Amplitude Divergence Detection
    # Directional volume pressure
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] < df['open']).astype(int)
    
    # Rolling volume pressure metrics
    df['up_volume_pressure'] = df['volume'] * df['up_day'].rolling(window=3).mean()
    df['down_volume_pressure'] = df['volume'] * df['down_day'].rolling(window=3).mean()
    df['volume_pressure_ratio'] = df['up_volume_pressure'] / (df['down_volume_pressure'] + 1e-8)
    
    # Amplitude efficiency (average from t-2 to t)
    df['amplitude_efficiency_ma'] = df['amplitude_efficiency'].rolling(window=3).mean()
    
    # Divergence strength (rolling correlation between amplitude asymmetry and volume)
    df['amplitude_asymmetry'] = (df['high'] - df['close'].shift(1)) / (df['close'].shift(1) - df['low'])
    df['amplitude_asymmetry'] = df['amplitude_asymmetry'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate rolling correlation safely
    def safe_corr(x, y):
        if len(x) < 3 or len(y) < 3:
            return np.nan
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 3:
            return np.nan
        return np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
    
    df['divergence_strength'] = pd.Series([
        safe_corr(
            df['amplitude_asymmetry'].iloc[max(0, i-4):i+1].values,
            df['volume'].iloc[max(0, i-4):i+1].values
        ) if i >= 4 else np.nan
        for i in range(len(df))
    ], index=df.index)
    
    # Regime-Adaptive Signal Processing
    # Define regime thresholds
    high_amplitude_threshold = df['amplitude_regime'].quantile(0.7)
    low_amplitude_threshold = df['amplitude_regime'].quantile(0.3)
    
    # Initialize regime signals
    df['regime_signal'] = 0
    
    # High amplitude regime: focus on extreme divergences and reversals
    high_amp_mask = df['amplitude_regime'] > high_amplitude_threshold
    df.loc[high_amp_mask, 'regime_signal'] = (
        df.loc[high_amp_mask, 'divergence_strength'] * 
        df.loc[high_amp_mask, 'volume_pressure_ratio'] * 
        (1 - df.loc[high_amp_mask, 'amplitude_efficiency_ma'])
    )
    
    # Low amplitude regime: emphasize volume persistence and accumulation
    low_amp_mask = df['amplitude_regime'] < low_amplitude_threshold
    df.loc[low_amp_mask, 'regime_signal'] = (
        df.loc[low_amp_mask, 'volume_pressure_ratio'] * 
        df.loc[low_amp_mask, 'amplitude_efficiency_ma'] * 
        np.sign(df.loc[low_amp_mask, 'divergence_strength'])
    )
    
    # Transition regime: balance divergence and confirmation signals
    transition_mask = ~(high_amp_mask | low_amp_mask)
    df.loc[transition_mask, 'regime_signal'] = (
        df.loc[transition_mask, 'divergence_strength'] * 
        df.loc[transition_mask, 'amplitude_efficiency_ma'] * 
        np.log1p(df.loc[transition_mask, 'volume_pressure_ratio'])
    )
    
    # Composite Factor Generation
    # Volume-confirmed amplitude signals
    df['volume_confirmed_amplitude'] = (
        df['short_term_amplitude'] * 
        np.sign(df['volume_pressure_ratio'] - 1) * 
        df['amplitude_efficiency_ma']
    )
    
    # Regime-weighted divergence scores
    df['regime_weight'] = np.where(
        high_amp_mask, 1.5,
        np.where(low_amp_mask, 0.8, 1.0)
    )
    
    df['weighted_divergence'] = df['divergence_strength'] * df['regime_weight']
    
    # Adaptive predictive signals based on regime state
    df['composite_factor'] = (
        df['regime_signal'] * 0.4 +
        df['volume_confirmed_amplitude'] * 0.3 +
        df['weighted_divergence'] * 0.3
    )
    
    # Final normalization
    factor = df['composite_factor'].fillna(0)
    
    return factor
