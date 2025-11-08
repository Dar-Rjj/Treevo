import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Timeframe Volatility Fractals
    # True Range calculation
    data['TR'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    
    # Short-term volatility pattern (3-day)
    data['vol_clustering'] = data['TR'] / data['TR'].shift(1)
    data['vol_regime_3d'] = data['vol_clustering'].rolling(window=3, min_periods=1).mean()
    
    # Medium-term volatility pattern (8-day)
    data['avg_TR_8d'] = data['TR'].rolling(window=8, min_periods=1).mean()
    data['avg_TR_16d'] = data['TR'].rolling(window=16, min_periods=1).mean()
    data['vol_momentum'] = (data['avg_TR_8d'] - data['avg_TR_16d']) / (data['avg_TR_16d'] + 1e-8)
    
    # Volatility fractal dimension
    data['vol_fractal_ratio'] = data['vol_regime_3d'] / (data['avg_TR_8d'] + 1e-8)
    
    # Fractal persistence calculation
    data['vol_fractal_dir'] = np.where(data['vol_fractal_ratio'] > data['vol_fractal_ratio'].shift(1), 1, -1)
    data['vol_persistence'] = 0
    for i in range(1, len(data)):
        if data['vol_fractal_dir'].iloc[i] == data['vol_fractal_dir'].iloc[i-1]:
            data.loc[data.index[i], 'vol_persistence'] = data['vol_persistence'].iloc[i-1] + 1
    
    data['fractal_decay'] = 0.95 ** data['vol_persistence']
    
    # 2. Price Efficiency Fractal Analysis
    # Intraday efficiency patterns
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['hl_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_momentum'] = data['intraday_efficiency'] - data['intraday_efficiency'].shift(3)
    
    # Multi-day efficiency convergence
    data['eff_3d_avg'] = data['intraday_efficiency'].rolling(window=3, min_periods=1).mean()
    data['eff_5d_avg'] = data['intraday_efficiency'].rolling(window=5, min_periods=1).mean()
    data['eff_divergence'] = data['intraday_efficiency'] - data['eff_5d_avg']
    
    # Fractal Efficiency Score
    data['fractal_efficiency'] = (data['intraday_efficiency'] * data['hl_utilization'] * 
                                 data['vol_fractal_ratio'] * data['fractal_decay'])
    
    # 3. Volume Fractal Structure
    data['volume_velocity'] = data['volume'] / (data['volume'].shift(1) + 1e-8) - 1
    data['volume_acceleration'] = data['volume_velocity'] - data['volume_velocity'].shift(1)
    
    # Volume-price fractal correlation
    data['vol_price_corr'] = data['volume_velocity'].rolling(window=5, min_periods=1).corr(data['intraday_efficiency'])
    data['corr_persistence'] = 0
    for i in range(1, len(data)):
        if (data['vol_price_corr'].iloc[i] > 0 and data['vol_price_corr'].iloc[i-1] > 0) or \
           (data['vol_price_corr'].iloc[i] < 0 and data['vol_price_corr'].iloc[i-1] < 0):
            data.loc[data.index[i], 'corr_persistence'] = data['corr_persistence'].iloc[i-1] + 1
    
    data['volume_fractal_score'] = (data['volume_velocity'] * data['volume_acceleration'] * 
                                   abs(data['vol_price_corr']) * (1 + 0.1 * data['corr_persistence']))
    
    # 4. Regime-Switching Momentum Framework
    # Price momentum regimes
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_acceleration'] = data['momentum_5d'] - data['momentum_10d']
    
    # Volatility-adjusted momentum
    data['vol_adj_momentum'] = data['momentum_5d'] / (data['avg_TR_8d'] + 1e-8)
    data['momentum_efficiency'] = data['momentum_5d'] / (data['TR'] + 1e-8)
    
    # Multi-timeframe momentum convergence
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_convergence'] = data['momentum_3d'] / (data['momentum_8d'] + 1e-8)
    
    # 5. Fractal Microstructure Integration
    # Amount-based flow patterns
    data['amount_velocity'] = data['amount'] / (data['amount'].shift(1) + 1e-8) - 1
    data['vol_amount_corr'] = data['volume'].rolling(window=5, min_periods=1).corr(data['amount'])
    
    # Price impact fractal measurement
    data['price_impact'] = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    data['vol_adj_impact'] = data['price_impact'] / (data['TR'] + 1e-8)
    
    data['fractal_impact_score'] = (data['vol_adj_impact'] * data['intraday_efficiency'] * 
                                   abs(data['vol_amount_corr']))
    
    # 6. Pattern Persistence and Regime Quality
    # Efficiency pattern persistence
    data['eff_persistence'] = 0
    for i in range(1, len(data)):
        if (data['intraday_efficiency'].iloc[i] > data['eff_5d_avg'].iloc[i] and 
            data['intraday_efficiency'].iloc[i-1] > data['eff_5d_avg'].iloc[i-1]) or \
           (data['intraday_efficiency'].iloc[i] < data['eff_5d_avg'].iloc[i] and 
            data['intraday_efficiency'].iloc[i-1] < data['eff_5d_avg'].iloc[i-1]):
            data.loc[data.index[i], 'eff_persistence'] = data['eff_persistence'].iloc[i-1] + 1
    
    # Volume pattern persistence
    data['vol_persistence'] = 0
    for i in range(1, len(data)):
        if (data['volume_velocity'].iloc[i] > 0 and data['volume_velocity'].iloc[i-1] > 0) or \
           (data['volume_velocity'].iloc[i] < 0 and data['volume_velocity'].iloc[i-1] < 0):
            data.loc[data.index[i], 'vol_persistence'] = data['vol_persistence'].iloc[i-1] + 1
    
    # Regime quality assessment
    data['vol_regime_quality'] = 1 / (1 + abs(data['vol_momentum']))
    data['eff_regime_quality'] = 1 / (1 + abs(data['eff_divergence']))
    data['momentum_regime_quality'] = 1 / (1 + abs(data['momentum_convergence'] - 1))
    
    # Pattern quality composite
    data['pattern_quality'] = (data['vol_regime_quality'] * data['eff_regime_quality'] * 
                              data['momentum_regime_quality'] * 
                              (1 + 0.05 * (data['eff_persistence'] + data['vol_persistence'])))
    
    # 7. Final Alpha Construction
    # Core fractal efficiency component
    core_fractal = (data['fractal_efficiency'] * data['volume_fractal_score'] * 
                   data['pattern_quality'])
    
    # Regime-adaptive momentum integration
    adaptive_momentum = (data['momentum_convergence'] * data['momentum_regime_quality'] * 
                        data['pattern_quality'])
    
    # Microstructure impact adjustment
    microstructure_impact = (data['fractal_impact_score'] * data['vol_amount_corr'] * 
                           data['pattern_quality'])
    
    # Final alpha output
    alpha = core_fractal * adaptive_momentum * microstructure_impact
    
    return alpha
