import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['return'] = (data['close'] - data['prev_close']) / data['prev_close']
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    
    # 1. Intraday Asymmetry
    # Overnight Gap Persistence
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_persistence'] = np.sign(data['gap']) * data['intraday_return']
    
    # Range Position Asymmetry
    data['range'] = data['high'] - data['low']
    data['position'] = ((data['close'] - data['low']) / data['range']) - 0.5
    
    # Volatility Asymmetry
    up_mask = data['close'] > data['prev_close']
    down_mask = data['close'] < data['prev_close']
    data['up_day_tr'] = np.where(up_mask, data['range'], np.nan)
    data['down_day_tr'] = np.where(down_mask, data['range'], np.nan)
    
    # 2. Volume-Price Asymmetry
    # Volume moving averages
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Directional Volume Bias
    data['up_day_volume'] = np.where(up_mask, data['volume'], np.nan)
    data['down_day_volume'] = np.where(down_mask, data['volume'], np.nan)
    
    # Volume-Volatility Mismatch
    data['volume_ratio_20'] = data['volume'] / data['volume_ma_20']
    high_volume_small_move = (data['volume_ratio_20'] > 1.5) & (abs(data['return']) < 0.01)
    low_volume_large_move = (data['volume_ratio_20'] < 0.7) & (abs(data['return']) > 0.03)
    data['volume_vol_mismatch'] = np.where(high_volume_small_move, -1, 
                                          np.where(low_volume_large_move, 1, 0))
    
    # 3. Regime-Dependent Asymmetry
    # Volatility regimes
    data['volatility_20d'] = data['return'].rolling(window=20, min_periods=10).std()
    vol_80_percentile = data['volatility_20d'].rolling(window=252, min_periods=63).quantile(0.8)
    vol_20_percentile = data['volatility_20d'].rolling(window=252, min_periods=63).quantile(0.2)
    
    high_vol_mask = data['volatility_20d'] > vol_80_percentile
    low_vol_mask = data['volatility_20d'] < vol_20_percentile
    data['high_vol_return'] = np.where(high_vol_mask, data['return'], np.nan)
    data['low_vol_return'] = np.where(low_vol_mask, data['return'], np.nan)
    
    # Trend-Regime Autocorrelation
    data['trend_strength'] = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else np.nan, raw=True
    )
    data['return_autocorr'] = data['return'].rolling(window=10, min_periods=5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else np.nan, raw=True
    )
    
    # Liquidity-Regime Price Impact
    data['high_liq_impact'] = abs(data['return']) / (data['volume'] / data['volume_ma_20']).replace(0, np.nan)
    data['low_liq_impact'] = abs(data['return']) * (data['volume'] / data['volume_ma_20'])
    
    # 4. Cross-Asymmetry Signals
    # Extreme Asymmetry Reversal
    gap_extreme = (abs(data['gap']) > 0.02) & (data['gap_persistence'] < -0.01)
    volume_extreme = (data['volume_ratio_20'] > 2) & (data['position'] > 0.8)
    data['extreme_asymmetry'] = np.where(gap_extreme | volume_extreme, -1, 0)
    
    # Multi-Timeframe Alignment
    data['prev_return'] = data['return'].shift(1)
    intraday_daily_align = np.sign(data['return']) == np.sign(data['prev_return'])
    
    data['volatility_5d'] = data['return'].rolling(window=5, min_periods=3).std()
    volatility_align = (data['volatility_5d'] / data['volatility_20d']) > 1.2
    
    data['multi_timeframe_align'] = np.where(intraday_daily_align & volatility_align, 1, 0)
    
    # Regime Transition Signals
    vol_ratio = data['volatility_5d'] / data['volatility_20d']
    data['vol_ratio_prev'] = vol_ratio.shift(1)
    volatility_breakout = (vol_ratio > 1.5) & (data['vol_ratio_prev'] <= 1.5)
    
    data['volume_ratio_prev'] = data['volume_ratio_20'].shift(1)
    liquidity_shift = (data['volume_ratio_20'] > 2.0) & (data['volume_ratio_prev'] <= 2.0)
    
    data['regime_transition'] = np.where(volatility_breakout | liquidity_shift, 1, 0)
    
    # Combine factors with weights
    factors = [
        data['gap_persistence'] * 0.15,
        data['position'] * 0.12,
        (data['up_day_tr'].fillna(0) - data['down_day_tr'].fillna(0)) * 0.10,
        data['volume_vol_mismatch'] * 0.08,
        (data['high_vol_return'].fillna(0) - data['low_vol_return'].fillna(0)) * 0.12,
        data['trend_strength'] * data['return_autocorr'] * 0.10,
        (data['high_liq_impact'] - data['low_liq_impact']) * 0.08,
        data['extreme_asymmetry'] * 0.10,
        data['multi_timeframe_align'] * 0.08,
        data['regime_transition'] * 0.07
    ]
    
    # Final alpha factor
    alpha_factor = sum(factors)
    
    return alpha_factor
