import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function to calculate True Range
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Gap-Momentum Divergence Analysis
    # Short-Term Gap Efficiency
    data['gap_efficiency_short'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_efficiency_short'] = data['gap_efficiency_short'].replace([np.inf, -np.inf], np.nan)
    
    # Medium-Term Gap Efficiency
    data['high_low_range'] = data['high'] - data['low']
    data['gap_efficiency_medium'] = np.abs(data['close'] - data['open'].shift(5)) / data['high_low_range'].rolling(window=6).sum()
    data['gap_efficiency_medium'] = data['gap_efficiency_medium'].replace([np.inf, -np.inf], np.nan)
    
    # Gap Efficiency Divergence
    data['gap_efficiency_divergence'] = data['gap_efficiency_short'] - data['gap_efficiency_medium']
    
    # Turnover-Momentum Divergence
    # Price Momentum Components
    data['price_momentum_5d'] = (data['close'] / data['close'].shift(4)) - 1
    data['price_momentum_10d'] = (data['close'] / data['close'].shift(9)) - 1
    data['momentum_difference'] = data['price_momentum_5d'] - data['price_momentum_10d']
    
    # Turnover Components
    data['turnover'] = data['volume'] * data['close']
    data['turnover_avg_5d'] = data['turnover'].rolling(window=5).mean()
    data['turnover_avg_15d'] = data['turnover'].rolling(window=15).mean()
    data['turnover_ratio'] = (data['turnover_avg_5d'] / data['turnover_avg_15d']) - 1
    
    # Turnover-Momentum Divergence
    data['turnover_momentum_divergence'] = data['momentum_difference'] * data['turnover_ratio']
    
    # Combine Gap and Momentum Divergence
    data['gap_momentum_divergence'] = np.sqrt(np.abs(data['gap_efficiency_divergence'] * data['turnover_momentum_divergence'])) * np.sign(data['gap_efficiency_divergence'] * data['turnover_momentum_divergence'])
    
    # Volume-Pressure Confirmation System
    # Gap Pressure Components
    data['morning_gap_pressure'] = (data['high'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['morning_gap_pressure'] = data['morning_gap_pressure'].replace([np.inf, -np.inf], np.nan)
    
    data['gap_fill_pressure'] = (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['gap_fill_pressure'] = data['gap_fill_pressure'].replace([np.inf, -np.inf], np.nan)
    
    data['pressure_asymmetry'] = data['morning_gap_pressure'] - data['gap_fill_pressure']
    
    # Volume Asymmetry Confirmation
    # Upside Volume Ratio
    data['is_up_day'] = (data['close'] > data['close'].shift(1)).astype(int)
    up_day_volume = data['volume'].rolling(window=10).apply(lambda x: x[data['is_up_day'].iloc[-10:].values == 1].mean() if (data['is_up_day'].iloc[-10:].values == 1).any() else 0)
    total_avg_volume = data['volume'].rolling(window=10).mean()
    data['upside_volume_ratio'] = up_day_volume / total_avg_volume
    
    # Price Movement Asymmetry
    data['returns'] = data['close'].pct_change()
    positive_returns_sum = data['returns'].rolling(window=10).apply(lambda x: x[x > 0].sum())
    negative_returns_sum = data['returns'].rolling(window=10).apply(lambda x: np.abs(x[x < 0]).sum())
    data['price_asymmetry'] = np.log1p(positive_returns_sum) - np.log1p(negative_returns_sum)
    
    # Combine Volume Asymmetry Components
    data['volume_asymmetry'] = np.log1p(np.abs(data['upside_volume_ratio'] * data['price_asymmetry'])) * np.sign(data['upside_volume_ratio'] * data['price_asymmetry'])
    
    # Integrate Pressure-Volume Confirmation
    data['pressure_volume_confirmation'] = np.cbrt(data['pressure_asymmetry'] * data['volume_asymmetry'])
    
    # Multi-Scale Volatility Regime Detection
    # Gap Volatility Compression
    data['gap_magnitude'] = np.abs(data['close'] - data['open'])
    data['gap_volatility_5d'] = data['gap_magnitude'].rolling(window=5).sum()
    data['gap_volatility_10d'] = data['gap_magnitude'].rolling(window=10).sum()
    data['gap_volatility_compression'] = (data['gap_volatility_5d'] / data['gap_volatility_10d']) - 1
    
    # True Range Volatility Component
    data['true_range'] = true_range(data['high'], data['low'], data['close'].shift(1))
    data['true_range_avg_20d'] = data['true_range'].rolling(window=20).mean()
    
    # High-Low Range Expansion
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    data['range_expansion'] = (data['high_20d'] / data['low_20d']) - 1
    
    # Volume Cluster Dynamics Enhancement
    # Gap Volume Fractal
    data['gap_volume'] = data['volume'] * (data['open'] != data['close'].shift(1)).astype(int)
    gap_volume_3d_range = data['gap_volume'].rolling(window=3).apply(lambda x: x.max() - x.min())
    gap_volume_8d_range = data['gap_volume'].rolling(window=8).apply(lambda x: x.max() - x.min())
    data['gap_volume_fractal'] = np.log(gap_volume_8d_range) / np.log(gap_volume_3d_range)
    data['gap_volume_fractal'] = data['gap_volume_fractal'].replace([np.inf, -np.inf], np.nan)
    
    # Gap Turnover Clusters
    data['gap_turnover'] = data['turnover'] * (data['open'] != data['close'].shift(1)).astype(int)
    gap_turnover_median_8d = data['gap_turnover'].rolling(window=8).median()
    data['gap_turnover_momentum'] = data['gap_turnover'] / data['gap_turnover'].rolling(window=4).apply(lambda x: x[:-1].max() if len(x[:-1]) > 0 else 1)
    
    # Gap cluster duration (simplified implementation)
    high_turnover = (data['gap_turnover'] > 2.5 * gap_turnover_median_8d).astype(int)
    data['gap_cluster_duration'] = high_turnover.rolling(window=5).sum()
    
    # Volume Cluster Multipliers
    data['volume_cluster_multiplier'] = (1 + data['gap_cluster_duration'] / 5) * data['gap_volume_fractal'].fillna(1) * data['gap_turnover_momentum'].fillna(1)
    
    # Final Alpha Synthesis
    # Combine Divergence and Confirmation Components
    divergence_confirmation = data['gap_momentum_divergence'] * data['pressure_volume_confirmation']
    intensity_scaled = divergence_confirmation * data['volume_cluster_multiplier']
    risk_adjusted = intensity_scaled / data['true_range_avg_20d'].replace(0, np.nan)
    
    # Apply Volatility Regime Filters
    volatility_scaled = risk_adjusted * data['range_expansion']
    regime_weighted = volatility_scaled * data['gap_volatility_compression']
    
    # Final alpha factor
    alpha = np.cbrt(regime_weighted)
    
    return alpha
