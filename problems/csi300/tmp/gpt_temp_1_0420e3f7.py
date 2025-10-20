import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Regime-Switching and Adaptive Efficiency
    """
    data = df.copy()
    
    # Multi-Timeframe Divergence Analysis
    # Price-Volume Divergence
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['divergence_5d'] = data['price_momentum_5d'] - data['volume_momentum_5d']
    data['divergence_persistence'] = data['divergence_5d'].rolling(window=3).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 3 else np.nan
    )
    
    # Acceleration Divergence
    data['price_acceleration'] = data['price_momentum_5d'] - data['price_momentum_5d'].shift(1)
    data['volume_acceleration'] = data['volume_momentum_5d'] - data['volume_momentum_5d'].shift(1)
    data['acceleration_divergence'] = data['price_acceleration'] - data['volume_acceleration']
    
    # Acceleration divergence reversals
    data['acc_div_reversal'] = (np.sign(data['acceleration_divergence']) != 
                               np.sign(data['acceleration_divergence'].shift(1))).astype(int)
    
    # Multi-Scale Divergence Integration
    data['divergence_strength'] = (
        data['divergence_5d'] * data['divergence_persistence'] + 
        data['acceleration_divergence'] * (1 + data['acc_div_reversal'])
    )
    
    # Market Regime Detection
    # Volatility Regime Classification
    data['daily_return'] = data['close'].pct_change()
    data['volatility_20d'] = data['daily_return'].rolling(window=20).std()
    data['volatility_median_60d'] = data['volatility_20d'].rolling(window=60).median()
    data['high_vol_regime'] = (data['volatility_20d'] > data['volatility_median_60d']).astype(int)
    data['vol_regime_persistence'] = data['high_vol_regime'].rolling(window=5).sum()
    
    # Trend Regime Identification
    data['trend_10d'] = data['close'] / data['close'].shift(10) - 1
    data['trend_30d'] = data['close'] / data['close'].shift(30) - 1
    data['trend_alignment'] = np.sign(data['trend_10d']) == np.sign(data['trend_30d'])
    data['trending_regime'] = (data['trend_alignment'] & 
                              (abs(data['trend_10d']) > 0.02)).astype(int)
    
    # Regime-Adaptive Signal Adjustment
    data['volatility_weight'] = np.where(data['high_vol_regime'] == 1, 0.7, 1.2)
    data['trend_weight'] = np.where(data['trending_regime'] == 1, 1.3, 0.8)
    data['regime_adjusted_divergence'] = (
        data['divergence_strength'] * data['volatility_weight'] * data['trend_weight']
    )
    
    # Adaptive Efficiency Measurement
    # Dynamic Range Efficiency
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['range_efficiency_5d_avg'] = data['range_efficiency'].rolling(window=5).mean()
    data['range_efficiency_deviation'] = data['range_efficiency'] - data['range_efficiency_5d_avg']
    
    # Volume Efficiency Analysis
    data['volume_efficiency'] = abs(data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    data['volume_efficiency_median_10d'] = data['volume_efficiency'].rolling(window=10).median()
    data['volume_efficiency_deviation'] = (
        data['volume_efficiency'] - data['volume_efficiency_median_10d']
    )
    
    # Multi-Dimensional Efficiency Integration
    data['composite_efficiency'] = (
        data['range_efficiency_deviation'] * 0.6 + 
        data['volume_efficiency_deviation'] * 0.4
    )
    
    # Pressure Accumulation with Regime Context
    # Regime-Weighted Pressure Calculation
    data['raw_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['pressure_vol_adjust'] = np.where(data['high_vol_regime'] == 1, 1.5, 1.0)
    data['pressure_trend_bias'] = np.where(data['trending_regime'] == 1, 
                                          np.sign(data['trend_10d']), 0)
    data['regime_contextual_pressure'] = (
        data['raw_pressure'] * data['pressure_vol_adjust'] + data['pressure_trend_bias'] * 0.1
    )
    
    # Multi-Timeframe Pressure Accumulation
    data['cumulative_pressure_3d'] = data['regime_contextual_pressure'].rolling(window=3).sum()
    data['pressure_divergence'] = (
        np.sign(data['cumulative_pressure_3d']) != np.sign(data['price_momentum_5d'])
    ).astype(int)
    
    # Composite Alpha Generation
    # Regime-Adaptive Signal Combination
    data['regime_confidence'] = (
        data['vol_regime_persistence'] / 5 * 0.4 + 
        data['trending_regime'] * 0.6
    )
    
    data['efficiency_filter'] = np.where(
        data['composite_efficiency'] > data['composite_efficiency'].rolling(window=10).quantile(0.7),
        1.2, 0.8
    )
    
    data['pressure_confirmation'] = np.where(
        data['pressure_divergence'] == 0,
        np.sign(data['cumulative_pressure_3d']) * 1.1,
        0.9
    )
    
    # Final Predictive Factor
    data['alpha_factor'] = (
        data['regime_adjusted_divergence'] * 
        data['regime_confidence'] * 
        data['efficiency_filter'] * 
        data['pressure_confirmation']
    )
    
    # Clean up and return
    alpha_series = data['alpha_factor'].copy()
    alpha_series.name = 'multi_timeframe_divergence_regime_adaptive'
    
    return alpha_series
