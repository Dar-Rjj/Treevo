import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate daily returns and directions
    df['ret'] = df['close'] / df['close'].shift(1) - 1
    df['up_day'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['down_day'] = (df['close'] < df['close'].shift(1)).astype(int)
    
    # Multi-Scale Asymmetric Range Analysis
    # Daily upside range
    df['upside_range'] = np.maximum(df['high'] - df['open'], df['high'] - df['close'].shift(1))
    df['upside_range'] = df['upside_range'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Daily downside range
    df['downside_range'] = np.maximum(df['open'] - df['low'], df['close'].shift(1) - df['low'])
    df['downside_range'] = df['downside_range'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Multi-day upside range momentum
    periods = [3, 5, 10]
    for period in periods:
        df[f'upside_momentum_{period}'] = df['upside_range'] / df['upside_range'].rolling(period).mean()
        df[f'downside_momentum_{period}'] = df['downside_range'] / df['downside_range'].rolling(period).mean()
    
    # Range Asymmetry Ratio
    df['range_asymmetry_3'] = df['upside_momentum_3'] / (df['downside_momentum_3'] + 1e-8)
    df['range_asymmetry_5'] = df['upside_momentum_5'] / (df['downside_momentum_5'] + 1e-8)
    df['range_asymmetry_10'] = df['upside_momentum_10'] / (df['downside_momentum_10'] + 1e-8)
    
    # Asymmetric Volume Acceleration
    # Volume acceleration calculations
    df['vol_acc_3'] = (df['volume'] / df['volume'].shift(3) - 1)
    df['vol_acc_8'] = (df['volume'] / df['volume'].shift(8) - 1)
    df['vol_acc_diff'] = df['vol_acc_3'] - df['vol_acc_8']
    
    # Up-day and down-day volume acceleration
    df['up_vol_acc'] = df['vol_acc_diff'] * df['up_day']
    df['down_vol_acc'] = df['vol_acc_diff'] * df['down_day']
    
    # Volume Acceleration Asymmetry
    df['vol_acc_asymmetry'] = df['up_vol_acc'] - df['down_vol_acc']
    
    # Range Efficiency Momentum with Asymmetry
    # Upside and downside efficiency
    df['hl_range'] = df['high'] - df['low']
    df['hl_range'] = df['hl_range'].replace(0, 1e-8)  # Avoid division by zero
    
    df['upside_efficiency'] = np.where(df['close'] > df['open'], 
                                      (df['close'] - df['open']) / df['hl_range'], 0)
    df['downside_efficiency'] = np.where(df['close'] < df['open'], 
                                        (df['open'] - df['close']) / df['hl_range'], 0)
    
    # Multi-day efficiency momentum
    for period in periods:
        df[f'upside_eff_mom_{period}'] = (df['upside_efficiency'] / 
                                         df['upside_efficiency'].rolling(period).mean())
        df[f'downside_eff_mom_{period}'] = (df['downside_efficiency'] / 
                                           df['downside_efficiency'].rolling(period).mean())
    
    # Asymmetric Efficiency Ratio
    df['eff_asymmetry_5'] = df['upside_eff_mom_5'] / (np.abs(df['downside_eff_mom_5']) + 1e-8)
    
    # Price-Volume Acceleration Divergence with Asymmetry
    # Price acceleration
    df['price_acc_3'] = (df['close'] / df['close'].shift(3) - 1)
    df['price_acc_8'] = (df['close'] / df['close'].shift(8) - 1)
    df['price_acc_diff'] = df['price_acc_3'] - df['price_acc_8']
    
    # Asymmetric price acceleration
    df['upside_price_acc'] = df['price_acc_diff'] * df['up_day']
    df['downside_price_acc'] = df['price_acc_diff'] * df['down_day']
    
    # Price acceleration asymmetry ratio
    df['price_acc_asymmetry'] = df['upside_price_acc'] / (np.abs(df['downside_price_acc']) + 1e-8)
    
    # Asymmetric Volume Acceleration Divergence
    df['up_divergence'] = np.sign(df['upside_price_acc']) * -np.sign(df['up_vol_acc']) * df['up_day']
    df['down_divergence'] = np.sign(df['downside_price_acc']) * -np.sign(df['down_vol_acc']) * df['down_day']
    df['acc_divergence'] = df['up_divergence'] + df['down_divergence']
    
    # Amount-Enhanced Efficiency Momentum
    # Amount momentum
    df['amount_2d_avg'] = df['amount'].rolling(3).mean()  # t-2 to t
    df['amount_5d_avg'] = df['amount'].rolling(6).mean()  # t-5 to t
    df['amount_momentum'] = (df['amount_2d_avg'] / df['amount_5d_avg']) - 1
    
    # Amount acceleration
    df['amount_acc_3'] = (df['amount'] / df['amount'].shift(3) - 1)
    df['amount_acc_8'] = (df['amount'] / df['amount'].shift(8) - 1)
    df['amount_acc_diff'] = df['amount_acc_3'] - df['amount_acc_8']
    
    # Amount-weighted range efficiency
    df['amount_weighted_eff'] = df['upside_eff_mom_5'] * (1 + df['amount_momentum'])
    
    # Intraday Momentum Quality with Asymmetry
    # Asymmetric Open-Close Strength
    df['up_strength'] = np.where(df['close'] > df['open'], 
                                (df['close'] - df['open']) / df['hl_range'], 0)
    df['down_strength'] = np.where(df['close'] < df['open'], 
                                  (df['open'] - df['close']) / df['hl_range'], 0)
    
    # Directional Consistency
    df['price_dir'] = np.sign(df['close'] - df['close'].shift(1))
    df['price_dir_3d'] = np.sign(df['close'] - df['close'].shift(3))
    
    # Up-day consistency (3-day lookback)
    up_mask = df['up_day'] == 1
    df['up_consistency'] = 0
    for i in range(3, len(df)):
        if up_mask.iloc[i]:
            window = df['price_dir'].iloc[i-2:i+1]  # Current and previous 2 days
            target_dir = df['price_dir_3d'].iloc[i]
            df.loc[df.index[i], 'up_consistency'] = (window == target_dir).sum() / 3
    
    # Down-day consistency (3-day lookback)
    down_mask = df['down_day'] == 1
    df['down_consistency'] = 0
    for i in range(3, len(df)):
        if down_mask.iloc[i]:
            window = df['price_dir'].iloc[i-2:i+1]
            target_dir = df['price_dir_3d'].iloc[i]
            df.loc[df.index[i], 'down_consistency'] = (window == target_dir).sum() / 3
    
    # Asymmetric Momentum Quality
    df['up_momentum_quality'] = np.abs(df['up_strength']) * df['up_consistency']
    df['down_momentum_quality'] = np.abs(df['down_strength']) * df['down_consistency']
    df['momentum_quality'] = df['up_momentum_quality'] + df['down_momentum_quality']
    
    # Composite Asymmetric Alpha Factor
    # Core components
    df['core_range_asymmetry'] = (df['range_asymmetry_3'] + df['range_asymmetry_5'] + df['range_asymmetry_10']) / 3
    df['core_vol_acc_asymmetry'] = df['vol_acc_asymmetry']
    df['core_acc_divergence'] = df['acc_divergence']
    df['core_amount_enhanced'] = df['amount_weighted_eff']
    
    # Quality-weighted integration
    quality_weight = 1 + df['momentum_quality']
    
    # Final composite factor
    df['composite_alpha'] = (
        df['core_range_asymmetry'] * 0.25 +
        df['core_vol_acc_asymmetry'] * 0.25 +
        df['core_acc_divergence'] * 0.25 +
        df['core_amount_enhanced'] * 0.25
    ) * quality_weight
    
    # Clean up and return
    result = df['composite_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
