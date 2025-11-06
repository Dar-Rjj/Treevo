import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Gap Momentum with Volume Efficiency Confirmation
    """
    data = df.copy()
    
    # Calculate Overnight Gap Momentum
    data['gap_ratio'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_momentum'] = data['gap_ratio'] * np.sign(data['close'] - data['close'].shift(1))
    
    # Calculate True Range and Volatility
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['volatility_20d'] = data['true_range'].rolling(window=20, min_periods=10).std()
    
    # Identify Market Regime
    data['volatility_percentile'] = data['volatility_20d'].rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)) if len(x.dropna()) >= 30 else np.nan
    )
    data['low_vol_percentile'] = data['volatility_20d'].rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] < np.percentile(x, 20)) if len(x.dropna()) >= 30 else np.nan
    )
    
    # Calculate returns for regime-specific conditions
    data['return_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['return_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Apply Regime-Specific Momentum Weighting
    data['regime_adaptive_momentum'] = data['gap_momentum'].copy()
    
    # High Volatility Regime adjustments
    high_vol_mask = data['volatility_percentile'] == True
    data.loc[high_vol_mask, 'regime_adaptive_momentum'] = data.loc[high_vol_mask, 'gap_momentum'] * (
        1 + abs(data.loc[high_vol_mask, 'return_3d'])
    )
    
    # Low Volatility Regime adjustments
    low_vol_mask = data['low_vol_percentile'] == True
    momentum_breakdown = (data['return_5d'] > 0) & (data['return_3d'] < 0)
    data.loc[low_vol_mask & momentum_breakdown, 'regime_adaptive_momentum'] = (
        data.loc[low_vol_mask & momentum_breakdown, 'gap_momentum'] * 0.7
    )
    
    # Volume Efficiency Analysis
    data['volume_median_20d'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_ratio'] = data['volume'] / data['volume_median_20d']
    
    # Volume Persistence
    data['above_median_volume'] = (data['volume'] > data['volume_median_20d']).astype(int)
    data['volume_consistency'] = data['above_median_volume'].rolling(window=5, min_periods=3).mean()
    
    # Price-Volume Efficiency
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volume_efficiency'] = data['volume'] / data['daily_range'].replace(0, np.nan)
    data['efficiency_avg_10d'] = data['volume_efficiency'].rolling(window=10, min_periods=5).mean()
    data['efficiency_divergence'] = data['volume_efficiency'] / data['efficiency_avg_10d']
    
    # Generate Composite Alpha Signal
    data['composite_alpha'] = data['regime_adaptive_momentum'].copy()
    
    # High Volatility Regime Signal
    high_vol_signal = (
        data['regime_adaptive_momentum'] * 
        data['volume_ratio'] * 
        data['volume_consistency'] * 
        data['efficiency_divergence'] * 1.5
    )
    data.loc[high_vol_mask, 'composite_alpha'] = high_vol_signal.loc[high_vol_mask]
    
    # Low Volatility Regime Signal
    low_vol_signal = (
        data['regime_adaptive_momentum'] * 
        data['volume_consistency'] * 
        data['volume_ratio'] * 
        data['efficiency_divergence'] * 0.7
    )
    low_vol_signal_smoothed = low_vol_signal.rolling(window=3, min_periods=2).mean()
    data.loc[low_vol_mask, 'composite_alpha'] = low_vol_signal_smoothed.loc[low_vol_mask]
    
    # Cap extreme values
    alpha_std_20d = data['composite_alpha'].rolling(window=20, min_periods=10).std()
    data['final_alpha'] = data['composite_alpha'].clip(
        lower=-3 * alpha_std_20d,
        upper=3 * alpha_std_20d
    )
    
    return data['final_alpha']
