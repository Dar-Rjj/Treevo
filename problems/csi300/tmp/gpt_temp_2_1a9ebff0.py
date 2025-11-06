import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe volatility-volume convergence,
    range-pressure momentum efficiency, acceleration-volume fractal divergence,
    and opening momentum volume confluence.
    """
    data = df.copy()
    
    # Multi-Timeframe Volatility-Volume Convergence
    # True Range calculation
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Volatility Regime Assessment
    data['vol_short'] = data['TR'].rolling(window=5, min_periods=3).std()
    data['vol_medium'] = data['TR'].rolling(window=10, min_periods=5).std()
    data['vol_ratio'] = data['vol_short'] / data['vol_medium']
    
    # Multi-Scale Convergence Analysis
    # Short-term (5-day) convergence
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['convergence_5d'] = np.sign(data['price_momentum_5d'] - data['volume_momentum_5d'])
    
    # Medium-term (10-day) convergence
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['convergence_10d'] = np.sign(data['price_momentum_10d'] - data['volume_momentum_10d'])
    
    # Regime-Adaptive Signal Generation
    data['base_convergence'] = 0.6 * data['convergence_5d'] + 0.4 * data['convergence_10d']
    data['vol_convergence_factor'] = data['base_convergence'] * (1 + data['vol_ratio'])
    
    # Range-Pressure Momentum Efficiency
    # Dynamic Range Analysis
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['range_volatility'] = (data['high'] - data['low']).rolling(window=8, min_periods=4).std()
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    
    # Pressure Imbalance Analysis
    data['upward_pressure'] = (data['close'] - data['open']).rolling(window=3, min_periods=2).mean()
    data['downward_pressure'] = (data['open'] - data['close']).rolling(window=3, min_periods=2).mean()
    data['pressure_direction'] = np.sign(data['close'] - data['open'])
    data['pressure_persistence'] = data['pressure_direction'].rolling(window=3, min_periods=2).apply(
        lambda x: len(set(x)) == 1 if len(x) == 3 else np.nan
    )
    
    # Efficiency Signal Construction
    data['upside_efficiency'] = data['range_utilization'] * data['upward_pressure'] * data['pressure_persistence']
    data['downside_efficiency'] = (1 - data['range_utilization']) * data['downward_pressure'] * data['pressure_persistence']
    data['efficiency_factor'] = (data['upside_efficiency'] - data['downside_efficiency']) * data['range_expansion']
    
    # Acceleration-Volume Fractal Divergence
    # Multi-Timeframe Acceleration
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['accel_short'] = data['momentum_3d'] - data['momentum_3d'].shift(1)
    data['accel_medium'] = data['momentum_8d'] - data['momentum_8d'].shift(1)
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_weighted_accel'] = data['accel_short'] * data['volume_momentum_3d']
    
    # Fractal Divergence Detection
    data['accel_divergence'] = data['accel_short'] - data['accel_medium']
    data['volume_clustering'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-8)
    )
    
    # Divergence Signal Generation
    data['divergence_signal'] = data['accel_divergence'] * data['volume_clustering']
    data['divergence_factor'] = np.cbrt(data['divergence_signal'])
    
    # Opening Momentum Volume Confluence
    # Opening Momentum Analysis
    data['opening_gap'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_persistence'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    data['momentum_retention'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume Confluence Assessment
    data['volume_median_5d'] = data['volume'].rolling(window=5, min_periods=3).median()
    data['volume_momentum_conf'] = data['volume'] / data['volume_median_5d']
    
    # Volume-pressure correlation (6-day window)
    data['volume_pressure_corr'] = data['close'].rolling(window=6, min_periods=4).corr(data['volume'])
    
    # Confluence Signal Construction
    data['confluence_strength'] = data['momentum_retention'] * data['volume_momentum_conf']
    data['confluence_alignment'] = data['intraday_persistence'] * data['volume_pressure_corr']
    data['confluence_factor'] = data['confluence_strength'] * data['confluence_alignment']
    
    # Final factor combination with equal weights
    data['final_factor'] = (
        0.25 * data['vol_convergence_factor'] +
        0.25 * data['efficiency_factor'] +
        0.25 * data['divergence_factor'] +
        0.25 * data['confluence_factor']
    )
    
    # Clean up intermediate columns and return final factor
    result = data['final_factor'].copy()
    return result
