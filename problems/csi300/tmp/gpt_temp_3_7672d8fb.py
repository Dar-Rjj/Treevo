import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Acceleration Component
    data['momentum_5d'] = data['close'] - data['close'].shift(5)
    data['momentum_5d_change'] = data['momentum_5d'] - data['momentum_5d'].shift(5)
    
    # 3-day Momentum Slope using linear regression
    def momentum_slope_3d(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(3)
        y = series.values
        return np.polyfit(x, y, 1)[0]
    
    data['momentum_3d_slope'] = data['close'].rolling(window=3, min_periods=3).apply(
        momentum_slope_3d, raw=False
    )
    
    # Volume Acceleration Component
    data['volume_5d_change'] = data['volume'] - data['volume'].shift(5)
    
    def volume_slope_3d(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(3)
        y = series.values
        return np.polyfit(x, y, 1)[0]
    
    data['volume_3d_slope'] = data['volume'].rolling(window=3, min_periods=3).apply(
        volume_slope_3d, raw=False
    )
    
    # Acceleration Regime Classification
    data['price_acceleration'] = (data['momentum_5d_change'] > 0).astype(int) + (data['momentum_3d_slope'] > 0).astype(int)
    data['volume_acceleration'] = (data['volume_5d_change'] > 0).astype(int) + (data['volume_3d_slope'] > 0).astype(int)
    data['high_acceleration'] = ((data['price_acceleration'] == 2) & (data['volume_acceleration'] == 2)).astype(int)
    
    # Volatility-Adjusted Gap Analysis
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['gap_filling_efficiency'] = np.abs(data['close'] - data['close'].shift(1)) / (np.abs(data['open'] - data['close'].shift(1)) + 1e-8)
    
    # True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = np.abs(data['high'] - data['close'].shift(1))
    data['tr3'] = np.abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility measures
    data['vol_5d'] = data['close'].rolling(window=5, min_periods=5).std()
    data['vol_10d'] = data['close'].rolling(window=10, min_periods=10).std()
    
    # Volatility-weighted gap signal
    data['vol_weighted_gap'] = data['gap_filling_efficiency'] / (data['true_range'] + 1e-8)
    data['vol_persistence_ratio'] = data['vol_5d'] / (data['vol_10d'] + 1e-8)
    data['gap_signal'] = data['vol_weighted_gap'] * data['vol_persistence_ratio']
    
    # Price-Volume Harmony Detection
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['directional_alignment'] = data['price_direction'] * data['volume_direction']
    
    # Magnitude Consistency - Rank correlation over 5-day window
    def rank_correlation_5d(price_changes, volume_changes):
        if len(price_changes) < 5:
            return np.nan
        price_ranks = price_changes.rank()
        volume_ranks = volume_changes.rank()
        return price_ranks.corr(volume_ranks)
    
    data['price_change_mag'] = np.abs(data['close'] - data['close'].shift(1))
    data['volume_change_mag'] = np.abs(data['volume'] - data['volume'].shift(1))
    
    data['magnitude_consistency'] = data['price_change_mag'].rolling(window=5, min_periods=5).apply(
        lambda x: rank_correlation_5d(x, data['volume_change_mag'].loc[x.index]), raw=False
    )
    
    # Volatility-Adjusted Returns
    data['vol_adj_return_5d'] = (data['close'] - data['close'].shift(5)) / (data['vol_5d'] + 1e-8)
    data['vol_adj_return_10d'] = (data['close'] - data['close'].shift(10)) / (data['vol_10d'] + 1e-8)
    
    # Microstructure Pressure Confirmation
    data['buying_pressure'] = (data['close'] - data['low']) - (data['high'] - data['close'])
    data['normalized_pressure'] = data['buying_pressure'] / (data['high'] - data['low'] + 1e-8)
    data['volume_weighted_pressure'] = data['normalized_pressure'] * data['volume']
    
    # Pressure Coherence Analysis
    data['pressure_imbalance'] = (data['normalized_pressure'] > 0).astype(int)
    data['consecutive_pressure_days'] = data['pressure_imbalance'].rolling(window=5, min_periods=1).apply(
        lambda x: x.iloc[-1] * x.sum() if x.iloc[-1] > 0 else 0, raw=False
    )
    data['cumulative_pressure_5d'] = data['normalized_pressure'].rolling(window=5, min_periods=5).sum()
    
    def pressure_trend_5d(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(5)
        y = series.values
        return np.polyfit(x, y, 1)[0]
    
    data['pressure_trend'] = data['normalized_pressure'].rolling(window=5, min_periods=5).apply(
        pressure_trend_5d, raw=False
    )
    
    # Adaptive Factor Construction
    data['regime_multiplier'] = np.where(data['high_acceleration'] == 1, 1.5, 0.8)
    
    # Harmony-Adjusted Momentum components
    data['gap_alignment_component'] = data['gap_signal'] * data['directional_alignment']
    data['magnitude_return_component'] = data['magnitude_consistency'] * data['vol_adj_return_5d']
    data['pressure_coherence'] = data['consecutive_pressure_days'] * data['pressure_trend']
    
    # Final Alpha Signal
    data['harmony_adjusted_momentum'] = (
        data['gap_alignment_component'] + 
        data['magnitude_return_component'] + 
        data['pressure_coherence']
    )
    
    data['final_alpha'] = data['regime_multiplier'] * data['harmony_adjusted_momentum'] * data['volume_weighted_pressure']
    
    return data['final_alpha']
