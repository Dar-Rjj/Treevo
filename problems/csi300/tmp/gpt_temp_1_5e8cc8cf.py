import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Momentum Divergence
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['vol_ma_5d'] = df['volume'].rolling(5).mean()
    df['vol_ma_10d'] = df['volume'].rolling(10).mean()
    df['vol_trend_slope'] = (df['vol_ma_5d'] - df['vol_ma_10d']) / df['vol_ma_10d']
    
    # Divergence signals
    df['price_down_vol_up'] = ((df['ret_5d'] < 0) & (df['vol_trend_slope'] > 0)).astype(int)
    df['price_up_vol_down'] = ((df['ret_5d'] > 0) & (df['vol_trend_slope'] < 0)).astype(int)
    df['momentum_vol_alignment'] = np.sign(df['ret_5d']) * df['vol_trend_slope']
    
    # Range Efficiency Breakout
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['abs_return'] = abs(df['close'].pct_change())
    df['efficiency_ratio'] = df['abs_return'] / df['true_range']
    df['efficiency_ratio'] = df['efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Breakout signals
    df['high_efficiency_breakout'] = ((df['efficiency_ratio'] > 0.7) & (df['close'] > df['prev_close'])).astype(int)
    df['low_efficiency_consolidation'] = (df['efficiency_ratio'] < 0.3).astype(int)
    df['range_expansion'] = (df['true_range'] > df['true_range'].rolling(5).mean()).astype(int)
    
    # Volume-Confirmed Reversal
    df['ret_3d_max'] = df['close'].pct_change(3).abs().rolling(3).max()
    df['ret_5d_max'] = df['close'].pct_change(5).abs().rolling(5).max()
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
    df['abnormal_move'] = (df['ret_3d_max'] > 2 * df['volatility_20d']).astype(int)
    
    df['volume_vs_20d_avg'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_spike'] = (df['volume_vs_20d_avg'] > 1.5).astype(int)
    df['volume_trend'] = df['volume'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Reversal signals
    df['overbought_reversal'] = ((df['ret_3d_max'] > 0) & (df['volume_spike'] == 1) & 
                                (df['close'] < df['prev_close'])).astype(int)
    df['oversold_reversal'] = ((df['ret_3d_max'] < 0) & (df['volume_spike'] == 1) & 
                              (df['close'] > df['prev_close'])).astype(int)
    df['reversal_strength'] = df['volume_vs_20d_avg'] * df['ret_3d_max'].abs()
    
    # Order Flow Persistence
    df['up_day'] = (df['close'] > df['prev_close']).astype(int)
    df['down_day'] = (df['close'] < df['prev_close']).astype(int)
    df['neutral_day'] = (df['close'] == df['prev_close']).astype(int)
    
    df['up_amount'] = df['amount'] * df['up_day']
    df['down_amount'] = df['amount'] * df['down_day']
    df['net_flow'] = df['up_amount'] - df['down_amount']
    
    df['flow_3d_sum'] = df['net_flow'].rolling(3).sum()
    df['flow_5d_sum'] = df['net_flow'].rolling(5).sum()
    
    # Consecutive flow direction
    df['flow_direction'] = np.sign(df['net_flow'])
    df['consecutive_flow_days'] = df['flow_direction'].groupby(
        (df['flow_direction'] != df['flow_direction'].shift()).cumsum()
    ).cumcount() + 1
    
    df['flow_acceleration'] = df['flow_3d_sum'].diff()
    
    # Volatility-Regime Volume Clustering
    df['atr_10d'] = df['true_range'].rolling(10).mean()
    df['atr_20d'] = df['true_range'].rolling(20).mean()
    df['volatility_ratio'] = df['atr_10d'] / df['atr_20d']
    
    df['volume_median_20d'] = df['volume'].rolling(20).median()
    df['high_volume_day'] = (df['volume'] > df['volume_median_20d']).astype(int)
    
    # Consecutive high volume days
    df['consecutive_high_vol'] = df['high_volume_day'].groupby(
        (df['high_volume_day'] != df['high_volume_day'].shift()).cumsum()
    ).cumcount() + 1
    df['volume_cluster_strength'] = df['volume'].rolling(5).sum() * df['consecutive_high_vol']
    
    # Regime-specific signals
    df['high_vol_high_volume'] = ((df['volatility_ratio'] > 1.1) & (df['high_volume_day'] == 1)).astype(int)
    df['low_vol_high_volume'] = ((df['volatility_ratio'] < 0.9) & (df['high_volume_day'] == 1)).astype(int)
    df['volume_breakout_rising_vol'] = ((df['volatility_ratio'] > 1) & 
                                       (df['volume'] > df['volume'].rolling(10).mean())).astype(int)
    
    # Composite alpha factor
    alpha = (
        df['momentum_vol_alignment'] * 0.15 +
        df['high_efficiency_breakout'] * 0.12 +
        df['reversal_strength'] * 0.18 +
        df['flow_5d_sum'] * 0.10 +
        df['volume_cluster_strength'] * 0.15 +
        df['volume_breakout_rising_vol'] * 0.10 +
        df['range_expansion'] * 0.08 +
        df['consecutive_flow_days'] * 0.12
    )
    
    return alpha
