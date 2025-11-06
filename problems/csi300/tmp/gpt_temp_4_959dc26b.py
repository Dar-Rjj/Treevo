import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Gap & Convergence Factor
    """
    data = df.copy()
    
    # Calculate daily returns for volatility estimation
    data['returns'] = data['close'].pct_change()
    
    # Multi-Timeframe Volatility Regime Classification
    # Short-term Volatility Estimation
    data['vol_5d'] = data['returns'].rolling(window=5).std()
    data['vol_momentum'] = data['vol_5d'] / data['vol_5d'].shift(10)
    
    # Regime Boundary Identification
    data['vol_percentile'] = data['vol_5d'].rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Classify volatility regimes
    conditions = [
        data['vol_percentile'] >= 0.7,
        data['vol_percentile'] <= 0.3,
        (data['vol_percentile'] > 0.3) & (data['vol_percentile'] < 0.7)
    ]
    choices = ['high', 'low', 'transitional']
    data['vol_regime'] = np.select(conditions, choices, default='transitional')
    
    # Volatility Persistence Analysis
    regime_changes = data['vol_regime'] != data['vol_regime'].shift(1)
    data['regime_counter'] = regime_changes.groupby((regime_changes).cumsum()).cumcount() + 1
    data['regime_stability'] = data['regime_counter'] / data['regime_counter'].rolling(window=20).max()
    
    # Gap Strength & Persistence Analysis
    data['gap_magnitude'] = np.abs(data['open'] / data['close'].shift(1) - 1)
    data['gap_direction'] = np.sign(data['open'] - data['close'].shift(1))
    
    # Gap direction consistency (3-day rolling)
    data['gap_dir_consistency'] = (
        data['gap_direction'].rolling(window=3).apply(
            lambda x: len(set(x)) == 1 if len(x) == 3 else 0, raw=False
        )
    )
    
    # Gap Persistence Component
    data['gap_streak'] = (
        (data['gap_direction'] == data['gap_direction'].shift(1)).astype(int)
    )
    data['gap_persistence'] = data['gap_streak'].rolling(window=3).sum() / 3
    
    # Gap Quality Assessment
    data['gap_vol_ratio'] = data['gap_magnitude'] / data['vol_5d']
    data['gap_quality'] = np.where(
        data['vol_regime'] == 'high',
        data['gap_vol_ratio'] * 0.8,
        data['gap_vol_ratio'] * 1.2
    )
    
    # Price-Volume Convergence Analysis
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_trend'] = data['price_efficiency'].rolling(window=5).mean()
    
    # Volume Concentration Analysis
    high_range_threshold = 0.8  # Top 20% of daily range
    low_range_threshold = 0.2   # Bottom 20% of daily range
    
    data['volume_high_side'] = data['volume'] * (
        (data['close'] - data['open'] * (1 - high_range_threshold)) / (data['high'] - data['low'] + 1e-8)
    ).clip(0, 1)
    
    data['volume_low_side'] = data['volume'] * (
        (data['open'] * (1 + low_range_threshold) - data['close']) / (data['high'] - data['low'] + 1e-8)
    ).clip(0, 1)
    
    data['volume_concentration'] = data['volume_high_side'] / (data['volume_low_side'] + 1e-8)
    
    # Convergence Signal Generation
    data['convergence_strength'] = (
        data['price_efficiency'] * data['volume_concentration'] * 
        np.sign(data['gap_direction'])
    )
    
    # Regime-Adaptive Signal Integration
    # High Volatility Regime Processing
    high_vol_mask = data['vol_regime'] == 'high'
    data.loc[high_vol_mask, 'regime_signal'] = (
        data['gap_magnitude'] * data['volume_concentration'] * data['vol_5d']
    )
    
    # Low Volatility Regime Processing
    low_vol_mask = data['vol_regime'] == 'low'
    data.loc[low_vol_mask, 'regime_signal'] = (
        data['gap_persistence'] * data['price_efficiency'] * (1 + data['regime_stability'])
    )
    
    # Transitional Regime Processing
    trans_mask = data['vol_regime'] == 'transitional'
    regime_persistence_weight = data['regime_stability'] * 0.5 + 0.5
    data.loc[trans_mask, 'regime_signal'] = (
        (data['gap_magnitude'] * data['volume_concentration'] * regime_persistence_weight) +
        (data['gap_persistence'] * data['price_efficiency'] * (1 - regime_persistence_weight))
    )
    
    # Multi-timeframe Signal Aggregation
    data['signal_5d'] = data['regime_signal'].rolling(window=5).mean()
    data['signal_10d'] = data['regime_signal'].rolling(window=10).mean()
    
    # Volatility-adjusted time decay
    vol_decay = 1 / (1 + data['vol_5d'] * 5)
    data['combined_signal'] = (
        data['signal_5d'] * 0.6 * vol_decay + 
        data['signal_10d'] * 0.4 * (1 - vol_decay)
    )
    
    # Final Alpha Factor Construction
    data['gap_convergence_score'] = (
        data['combined_signal'] * data['convergence_strength'] * 
        data['regime_stability'] * data['gap_quality']
    )
    
    # Signal Quality Enhancement
    volume_threshold = data['volume'].rolling(window=20).quantile(0.3)
    strong_signal_mask = (
        (data['volume'] > volume_threshold) & 
        (np.abs(data['gap_convergence_score']) > data['gap_convergence_score'].rolling(window=20).std())
    )
    
    data['final_alpha'] = data['gap_convergence_score']
    data.loc[strong_signal_mask, 'final_alpha'] = data['gap_convergence_score'] * 1.5
    
    # Apply regime-dependent confirmation filters
    high_vol_confirmation = (data['vol_regime'] == 'high') & (data['volume_concentration'] > 1.2)
    low_vol_confirmation = (data['vol_regime'] == 'low') & (data['gap_persistence'] > 0.6)
    
    data.loc[high_vol_confirmation, 'final_alpha'] *= 1.2
    data.loc[low_vol_confirmation, 'final_alpha'] *= 1.1
    
    return data['final_alpha']
