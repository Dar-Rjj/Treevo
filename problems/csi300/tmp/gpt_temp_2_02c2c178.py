import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Weighted Fractal Momentum with Volume Regime Divergence
    """
    data = df.copy()
    
    # Calculate basic price changes and volatility
    data['price_change'] = data['close'].diff()
    data['abs_price_change'] = np.abs(data['price_change'])
    
    # Multi-Timeframe Fractal Momentum Analysis
    # 3-day fractal momentum
    data['fractal_3d'] = (data['close'] - data['close'].shift(3)) / \
                         data['abs_price_change'].rolling(window=3, min_periods=1).sum()
    
    # 8-day fractal momentum
    data['fractal_8d'] = (data['close'] - data['close'].shift(8)) / \
                         data['abs_price_change'].rolling(window=8, min_periods=1).sum()
    
    # 21-day fractal momentum
    data['fractal_21d'] = (data['close'] - data['close'].shift(21)) / \
                          data['abs_price_change'].rolling(window=21, min_periods=1).sum()
    
    # Volume-Weighted Fractal Efficiency
    # Volume-adjusted price movement
    data['vol_weighted_move'] = data['volume'] * data['abs_price_change']
    
    # 3-day volume-weighted fractal
    data['vol_fractal_3d'] = data['vol_weighted_move'].rolling(window=3, min_periods=1).sum() / \
                             data['abs_price_change'].rolling(window=3, min_periods=1).sum()
    
    # 8-day volume-weighted fractal
    data['vol_fractal_8d'] = data['vol_weighted_move'].rolling(window=8, min_periods=1).sum() / \
                             data['abs_price_change'].rolling(window=8, min_periods=1).sum()
    
    # 21-day volume-weighted fractal
    data['vol_fractal_21d'] = data['vol_weighted_move'].rolling(window=21, min_periods=1).sum() / \
                              data['abs_price_change'].rolling(window=21, min_periods=1).sum()
    
    # Fractal efficiency divergence
    data['fractal_div_3d'] = data['fractal_3d'] - data['vol_fractal_3d']
    data['fractal_div_8d'] = data['fractal_8d'] - data['vol_fractal_8d']
    data['fractal_div_21d'] = data['fractal_21d'] - data['vol_fractal_21d']
    
    # Volatility-Adjusted Breakout Analysis
    # Multi-timeframe breakout detection
    data['high_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['breakout_3d'] = (data['close'] - data['high_3d']) / (data['high_3d'] - data['low_3d']).replace(0, np.nan)
    
    data['high_8d'] = data['high'].rolling(window=8, min_periods=1).max()
    data['low_8d'] = data['low'].rolling(window=8, min_periods=1).min()
    data['breakout_8d'] = (data['close'] - data['high_8d']) / (data['high_8d'] - data['low_8d']).replace(0, np.nan)
    
    data['high_21d'] = data['high'].rolling(window=21, min_periods=1).max()
    data['low_21d'] = data['low'].rolling(window=21, min_periods=1).min()
    data['breakout_21d'] = (data['close'] - data['high_21d']) / (data['high_21d'] - data['low_21d']).replace(0, np.nan)
    
    # Breakout consistency scoring
    data['breakout_aligned'] = ((data['breakout_3d'] > 0) & (data['breakout_8d'] > 0) & (data['breakout_21d'] > 0)).astype(int) - \
                              ((data['breakout_3d'] < 0) & (data['breakout_8d'] < 0) & (data['breakout_21d'] < 0)).astype(int)
    
    # Fractal-Breakout Divergence
    data['fractal_breakout_div_3d'] = data['fractal_3d'] * np.sign(data['breakout_3d'])
    data['fractal_breakout_div_8d'] = data['fractal_8d'] * np.sign(data['breakout_8d'])
    data['fractal_breakout_div_21d'] = data['fractal_21d'] * np.sign(data['breakout_21d'])
    
    # Intraday pressure
    data['intraday_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Adaptive Volume-Fractal Regime Classification
    # Volume efficiency analysis
    data['volume_change'] = data['volume'].diff()
    data['abs_volume_change'] = np.abs(data['volume_change'])
    
    data['vol_eff_3d'] = np.abs(data['volume'] - data['volume'].shift(3)) / \
                         data['abs_volume_change'].rolling(window=3, min_periods=1).sum()
    data['vol_eff_8d'] = np.abs(data['volume'] - data['volume'].shift(8)) / \
                         data['abs_volume_change'].rolling(window=8, min_periods=1).sum()
    data['vol_eff_21d'] = np.abs(data['volume'] - data['volume'].shift(21)) / \
                          data['abs_volume_change'].rolling(window=21, min_periods=1).sum()
    
    # Volume regime classification
    data['vol_regime'] = 0  # Medium by default
    data.loc[data['vol_eff_3d'] > 0.7, 'vol_regime'] = 1  # High volume fractal
    data.loc[data['vol_eff_3d'] < 0.3, 'vol_regime'] = -1  # Low volume fractal
    
    # Volume-Price Divergence Component
    # Volume momentum trends
    data['vol_trend_short'] = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    data['vol_trend_medium'] = data['volume'].rolling(window=15, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # Volume-price divergence
    data['vol_price_div'] = data['vol_trend_short'] * data['fractal_3d']
    
    # Multi-Dimensional Factor Synthesis with Exponential Weighting
    # Regime-adaptive weighting
    data['regime_weight_price'] = 0.5
    data['regime_weight_volume'] = 0.5
    
    # High volume regime
    high_vol_mask = data['vol_regime'] == 1
    data.loc[high_vol_mask, 'regime_weight_price'] = 0.4
    data.loc[high_vol_mask, 'regime_weight_volume'] = 0.6
    
    # Low volume regime
    low_vol_mask = data['vol_regime'] == -1
    data.loc[low_vol_mask, 'regime_weight_price'] = 0.7
    data.loc[low_vol_mask, 'regime_weight_volume'] = 0.3
    
    # Exponential weighting structure
    # Short-term components (3-day)
    short_term = (data['fractal_breakout_div_3d'] * 0.6 + 
                  data['fractal_div_3d'] * 0.4) * data['intraday_pressure']
    
    # Medium-term components (8-day)
    medium_term = (data['fractal_breakout_div_8d'] * 0.5 + 
                   data['fractal_div_8d'] * 0.5) * data['vol_price_div']
    
    # Long-term components (21-day)
    long_term = (data['fractal_breakout_div_21d'] * 0.4 + 
                 data['fractal_div_21d'] * 0.6) * data['breakout_aligned']
    
    # Combine with regime-adaptive weighting
    price_component = (short_term * 0.5 + medium_term * 0.3 + long_term * 0.2)
    volume_component = (data['vol_fractal_3d'] * 0.4 + data['vol_fractal_8d'] * 0.35 + data['vol_fractal_21d'] * 0.25)
    
    # Final factor synthesis
    alpha_factor = (price_component * data['regime_weight_price'] + 
                    volume_component * data['regime_weight_volume']) * \
                   (1 + data['vol_price_div'])
    
    # Non-linear signal enhancement
    # Amplify aligned signals, suppress contradictions
    aligned_mask = ((data['fractal_3d'] * data['fractal_8d'] > 0) & 
                    (data['fractal_8d'] * data['fractal_21d'] > 0))
    contradictory_mask = ((data['fractal_3d'] * data['fractal_8d'] < 0) | 
                          (data['fractal_8d'] * data['fractal_21d'] < 0))
    
    alpha_factor.loc[aligned_mask] *= 1.2
    alpha_factor.loc[contradictory_mask] *= 0.8
    
    # Clean and return
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_factor
