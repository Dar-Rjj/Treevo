import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Volatility Fractal Analysis
    # Volatility Regime Identification
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                             abs(data['low'] - data['close'].shift(1))))
    data['volatility_5d'] = data['true_range'].rolling(window=5).mean()
    
    # Volatility regime classification
    vol_median = data['volatility_5d'].rolling(window=20).median()
    data['vol_regime'] = np.where(data['volatility_5d'] > vol_median, 1, 0)  # 1=high, 0=low
    
    # Range compression
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Upside/Downside Volatility Fractal Asymmetry
    upside_vol = []
    downside_vol = []
    
    for i in range(len(data)):
        if i < 5:
            upside_vol.append(np.nan)
            downside_vol.append(np.nan)
            continue
            
        upside_sum = 0
        downside_sum = 0
        
        for j in range(5):
            idx = i - j
            if data['close'].iloc[idx] > data['close'].iloc[idx-1]:
                upside_sum += max(0, data['close'].iloc[idx] - data['close'].iloc[idx-1])
            elif data['close'].iloc[idx] < data['close'].iloc[idx-1]:
                downside_sum += abs(min(0, data['close'].iloc[idx] - data['close'].iloc[idx-1]))
        
        upside_vol.append(upside_sum)
        downside_vol.append(downside_sum)
    
    data['upside_vol_path'] = upside_vol
    data['downside_vol_path'] = downside_vol
    data['fractal_asymmetry'] = data['upside_vol_path'] / data['downside_vol_path']
    data['fractal_asymmetry'] = data['fractal_asymmetry'].replace([np.inf, -np.inf], np.nan)
    
    # Cross-Timeframe Fractal Dimension
    # 3-day volatility path and range
    vol_path_3d = data['close'].diff().abs().rolling(window=3).sum()
    price_range_3d = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min())
    data['fractal_3d'] = np.log(vol_path_3d) / np.log(price_range_3d)
    
    # 5-day volatility path and range
    vol_path_5d = data['close'].diff().abs().rolling(window=5).sum()
    price_range_5d = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    data['fractal_5d'] = np.log(vol_path_5d) / np.log(price_range_5d)
    
    data['fractal_timeframe_ratio'] = data['fractal_3d'] / data['fractal_5d']
    
    # Microstructure Momentum-Volume Fractal Dynamics
    # Multi-Timeframe Efficiency Fractals
    data['efficiency_1d'] = abs(data['close'] - data['open']) / data['volume']
    data['efficiency_5d'] = abs(data['close'] - data['open']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    data['efficiency_acceleration'] = data['efficiency_1d'] / data['efficiency_5d']
    
    # Volume-Momentum Fractal Alignment
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5)
    data['price_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['volume_price_fractal'] = data['volume_momentum'] / data['price_momentum']
    data['volume_price_fractal'] = data['volume_price_fractal'].replace([np.inf, -np.inf], np.nan)
    
    # Intraday Fractal Momentum Patterns
    data['opening_gap_persistence'] = (data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1))
    data['midday_price'] = (data['high'] + data['low']) / 2
    data['closing_momentum_fractal'] = (data['close'] - data['midday_price']) / abs(data['midday_price'] - data['open'])
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Fractal Divergence-Convergence Framework
    # Volatility-Momentum Fractal Divergence
    price_trend = np.sign(data['close'] - data['close'].shift(5))
    vol_trend = np.sign(data['fractal_asymmetry'] - data['fractal_asymmetry'].shift(1))
    data['vol_momentum_divergence'] = price_trend * vol_trend
    
    # Volume-Efficiency Fractal Convergence
    volume_breakout = (data['volume'] > data['volume'].rolling(window=10).mean() * 1.2).astype(int)
    efficiency_breakout = (data['efficiency_1d'] > data['efficiency_1d'].rolling(window=10).mean() * 1.2).astype(int)
    data['volume_efficiency_convergence'] = volume_breakout * efficiency_breakout
    
    # Cross-Timeframe Fractal Alignment
    fractal_consistency = (np.sign(data['fractal_3d'] - data['fractal_3d'].shift(1)) == 
                          np.sign(data['fractal_5d'] - data['fractal_5d'].shift(1))).astype(int)
    data['cross_timeframe_alignment'] = fractal_consistency
    
    # Regime-Adaptive Fractal Signal Integration
    # High Volatility Regime Signals
    high_vol_weight = np.where(data['vol_regime'] == 1, 
                              data['fractal_asymmetry'] * data['efficiency_acceleration'], 0)
    
    # Low Volatility Regime Signals  
    low_vol_weight = np.where(data['vol_regime'] == 0,
                             data['volume_price_fractal'] * data['intraday_efficiency'], 0)
    
    # Transition Detection
    regime_change = (data['vol_regime'] != data['vol_regime'].shift(1)).astype(int)
    transition_weight = regime_change * data['fractal_asymmetry'] * data['volume_efficiency_convergence']
    
    # Composite Fractal Factor Synthesis
    # Fractal Signal Strength Calculation
    volatility_component = data['fractal_asymmetry'] * data['volume_efficiency_convergence']
    volatility_component = volatility_component.fillna(0)
    
    # Scale by cross-timeframe alignment
    scaled_component = volatility_component * data['cross_timeframe_alignment']
    
    # Regime-adaptive weighting
    regime_weight = np.where(data['vol_regime'] == 1, 1.2, 0.8)
    
    # Final factor calculation
    factor = (scaled_component * regime_weight + 
             high_vol_weight + low_vol_weight + transition_weight)
    
    # Normalize and return
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor
