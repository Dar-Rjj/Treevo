import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Divergence with Fractal Efficiency and Regime-Switching Confirmation
    """
    data = df.copy()
    
    # Fractal Efficiency Calculation
    # Daily Price Movement Efficiency
    data['price_movement'] = np.abs(data['close'] - data['close'].shift(1))
    data['max_possible_movement'] = data['high'] - data['low']
    data['efficiency_ratio'] = data['price_movement'] / data['max_possible_movement']
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # 10-day Efficiency Variance (Fractal Complexity)
    data['efficiency_variance'] = data['efficiency_ratio'].rolling(window=10, min_periods=5).var()
    
    # Volume Fractal Analysis
    data['volume_movement'] = np.abs(data['volume'] - data['volume'].shift(1))
    data['volume_range_5d'] = data['volume'].rolling(window=5, min_periods=3).max() - data['volume'].rolling(window=5, min_periods=3).min()
    data['volume_efficiency'] = data['volume_movement'] / data['volume_range_5d']
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Efficiency Persistence
    volume_efficiency_high = (data['volume_efficiency'] > 0.5).astype(int)
    data['volume_persistence'] = volume_efficiency_high.groupby((volume_efficiency_high != volume_efficiency_high.shift(1)).cumsum()).cumcount() + 1
    
    # Price-Volume Divergence Detection
    # Directional Divergence
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['directional_divergence'] = data['price_direction'] * data['volume_direction']
    
    # Magnitude Divergence
    data['normalized_price_change'] = (data['close'] - data['close'].shift(1)) / data['max_possible_movement']
    data['normalized_volume_change'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    data['magnitude_divergence'] = np.abs(data['normalized_price_change']) - np.abs(data['normalized_volume_change'])
    
    # Time-Scale Divergence
    data['price_trend_3d'] = data['close'] / data['close'].shift(3) - 1
    data['price_trend_10d'] = data['close'] / data['close'].shift(10) - 1
    data['volume_trend_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_trend_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['time_scale_divergence'] = np.sign(data['price_trend_3d'] * data['price_trend_10d']) * np.abs(data['volume_trend_3d'] / (data['volume_trend_10d'] + 1e-8))
    
    # Regime-Switching Confirmation
    # Volatility Regime
    data['price_range_5d'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    data['price_range_20d_median'] = (data['high'] - data['low']).rolling(window=20, min_periods=10).median()
    data['volatility_regime'] = (data['price_range_5d'] > data['price_range_20d_median']).astype(int)
    
    # Trend Regime
    def linear_trend(x):
        if len(x) < 2:
            return 0
        return stats.linregress(range(len(x)), x)[0]
    
    data['price_slope_10d'] = data['close'].rolling(window=10, min_periods=5).apply(linear_trend, raw=True)
    data['trend_strength'] = np.abs(data['price_slope_10d']) / data['close'].rolling(window=10, min_periods=5).std()
    data['trend_regime'] = np.where(data['trend_strength'] > 0.1, np.sign(data['price_slope_10d']), 0)
    
    # Regime Persistence
    data['volatility_regime_persistence'] = data['volatility_regime'].groupby((data['volatility_regime'] != data['volatility_regime'].shift(1)).cumsum()).cumcount() + 1
    data['trend_regime_persistence'] = data['trend_regime'].groupby((data['trend_regime'] != data['trend_regime'].shift(1)).cumsum()).cumcount() + 1
    
    # Regime-Adaptive Parameters
    data['efficiency_threshold'] = np.where(data['volatility_regime'] == 1, 0.3, 0.6)
    data['divergence_threshold'] = np.where(data['volatility_regime'] == 1, 0.2, 0.1)
    data['regime_multiplier'] = np.where(data['volatility_regime_persistence'] >= 3, 1.2, 0.8)
    
    # Multi-Scale Confirmation Signals
    # Intraday Confirmation
    data['opening_gap'] = np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['intraday_confirmation'] = np.where(data['opening_gap'] < 0.02, 1.0, 0.7)
    
    # Short-term Momentum
    data['momentum_2d'] = data['close'] / data['close'].shift(2) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_alignment'] = np.sign(data['momentum_2d'] * data['momentum_5d'])
    data['momentum_confirmation'] = np.where(data['momentum_alignment'] > 0, 1.1, 0.9)
    
    # Medium-term Context
    data['ma_20'] = data['close'].rolling(window=20, min_periods=10).mean()
    data['position_vs_ma'] = data['close'] / data['ma_20'] - 1
    data['medium_term_context'] = np.where(np.abs(data['position_vs_ma']) < 0.1, 1.0, 0.8)
    
    # Alpha Factor Construction
    # Core Divergence Score
    data['core_divergence'] = (data['directional_divergence'] * data['magnitude_divergence'] * 
                              (1 - data['efficiency_variance']) * data['time_scale_divergence'])
    
    # Final Alpha Calculation with Confirmation Criteria
    efficiency_condition = data['efficiency_ratio'] > data['efficiency_threshold']
    volume_condition = data['volume_persistence'] >= 2
    divergence_condition = np.abs(data['core_divergence']) > data['divergence_threshold']
    regime_condition = data['volatility_regime_persistence'] >= 2
    
    data['alpha'] = 0.0
    valid_signals = efficiency_condition & volume_condition & divergence_condition & regime_condition
    
    data.loc[valid_signals, 'alpha'] = (
        data.loc[valid_signals, 'core_divergence'] * 
        data.loc[valid_signals, 'regime_multiplier'] * 
        data.loc[valid_signals, 'intraday_confirmation'] * 
        data.loc[valid_signals, 'momentum_confirmation'] * 
        data.loc[valid_signals, 'medium_term_context']
    )
    
    # Apply directional sign from price movement
    data['alpha'] = data['alpha'] * data['price_direction']
    
    return data['alpha']
