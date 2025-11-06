import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-scaled momentum reversal, 
    intraday strength persistence, price-volume efficiency, and range breakout quality.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Scaled Momentum Reversal
    # Multi-Timeframe Momentum Calculation
    data['ret_1d'] = data['close'].pct_change()
    data['mom_3d'] = data['close'].pct_change(periods=3)
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    data['mom_diff'] = data['mom_10d'] - data['mom_3d']
    
    # Dynamic Volatility Scaling
    data['range_vol'] = (data['high'] - data['low']) / data['close']
    data['range_vol_ma'] = data['range_vol'].rolling(window=5).mean()
    data['close_vol'] = data['ret_1d'].rolling(window=10).std()
    data['combined_vol'] = (data['range_vol_ma'] + data['close_vol']) / 2 + 0.0001
    
    # Volume-Weighted Adjustment
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_5']
    data['volume_price_align'] = data['volume_ratio'] * np.sign(data['ret_1d'])
    
    # Volatility-scaled momentum factor
    vol_scaled_mom = data['mom_diff'] / data['combined_vol'] * data['volume_price_align']
    
    # Intraday Strength Persistence Factor
    # Intraday Price Strength Calculation
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, 0.0001)
    data['rel_close_pos'] = (data['close'] - data['low']) / high_low_range
    data['intraday_efficiency'] = (data['close'] - data['open']) / high_low_range
    
    # Persistence Measurement
    data['strength_autocorr'] = data['rel_close_pos'].rolling(window=5).apply(
        lambda x: x.autocorr(), raw=False
    )
    
    # Trend consistency (count consecutive same direction moves)
    def count_consecutive_direction(series):
        if len(series) < 2:
            return 0
        direction_changes = (series.diff().fillna(0) > 0).astype(int)
        current_direction = direction_changes.iloc[-1]
        count = 1
        for i in range(len(direction_changes)-2, max(0, len(direction_changes)-11), -1):
            if direction_changes.iloc[i] == current_direction:
                count += 1
            else:
                break
        return count
    
    data['trend_consistency'] = data['rel_close_pos'].rolling(window=10).apply(
        count_consecutive_direction, raw=False
    )
    
    # Volume Confirmation
    data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    data['volume_surge'] = data['volume'] / data['volume_ma_10']
    
    # Volume-strength correlation
    data['volume_strength_corr'] = data['rel_close_pos'].rolling(window=10).corr(data['volume'])
    
    # Intraday strength factor
    intraday_factor = (data['strength_autocorr'] * data['trend_consistency'] * 
                      data['volume_surge'] * data['volume_strength_corr'] * 
                      data['rel_close_pos'])
    
    # Price-Volume Efficiency Factor
    # Volume-Weighted Price Impact
    data['return_per_volume'] = abs(data['ret_1d']) / (data['volume'] + 1)
    data['volume_adjusted_ret'] = data['ret_1d'] * data['volume_ratio']
    
    # Transaction Size Analysis
    data['avg_tx_size'] = data['amount'] / (data['volume'] + 1)
    data['avg_tx_size_ma'] = data['avg_tx_size'].rolling(window=10).mean()
    data['large_order_conc'] = data['avg_tx_size'] / data['avg_tx_size_ma']
    
    # Efficiency Persistence
    data['efficiency_ma'] = data['return_per_volume'].rolling(window=5).mean()
    data['vol_adj_efficiency'] = data['return_per_volume'] / (data['combined_vol'] + 0.0001)
    
    # Price-volume efficiency factor
    pv_efficiency_factor = (data['vol_adj_efficiency'] * data['efficiency_ma'] * 
                           data['large_order_conc'])
    
    # Range Breakout Quality Factor
    # Breakout Signal Generation
    data['high_5d'] = data['high'].rolling(window=5).max().shift(1)
    data['low_5d'] = data['low'].rolling(window=5).min().shift(1)
    data['high_10d'] = data['high'].rolling(window=10).max().shift(1)
    data['low_10d'] = data['low'].rolling(window=10).min().shift(1)
    
    # Multi-timeframe breakouts
    data['breakout_5d_up'] = (data['close'] > data['high_5d']).astype(int)
    data['breakout_5d_down'] = (data['close'] < data['low_5d']).astype(int)
    data['breakout_10d_up'] = (data['close'] > data['high_10d']).astype(int)
    data['breakout_10d_down'] = (data['close'] < data['low_10d']).astype(int)
    
    # Breakout magnitude
    data['breakout_mag_5d'] = np.where(data['breakout_5d_up'] == 1, 
                                      (data['close'] - data['high_5d']) / data['high_5d'],
                                      np.where(data['breakout_5d_down'] == 1,
                                              (data['low_5d'] - data['close']) / data['low_5d'], 0))
    
    data['breakout_mag_10d'] = np.where(data['breakout_10d_up'] == 1, 
                                       (data['close'] - data['high_10d']) / data['high_10d'],
                                       np.where(data['breakout_10d_down'] == 1,
                                               (data['low_10d'] - data['close']) / data['low_10d'], 0))
    
    # Volume confirmation for breakouts
    data['volume_surge_breakout'] = (data['volume'] > 1.2 * data['volume_ma_10']).astype(int)
    
    # Historical breakout performance
    data['fwd_ret_3d'] = data['close'].pct_change(periods=-3).shift(3)
    
    def breakout_success_rate(series):
        if len(series) < 20:
            return 0.5
        breakouts = series[series != 0]
        if len(breakouts) == 0:
            return 0.5
        successful = (breakouts > 0).sum()
        return successful / len(breakouts)
    
    data['breakout_success_rate'] = data['fwd_ret_3d'].rolling(window=20).apply(
        breakout_success_rate, raw=False
    )
    
    # Current breakout assessment
    data['breakout_score'] = ((data['breakout_mag_5d'] + data['breakout_mag_10d']) * 
                             data['volume_surge_breakout'])
    
    # Breakout quality factor
    breakout_factor = data['breakout_score'] * data['breakout_success_rate'] / (data['combined_vol'] + 0.0001)
    
    # Combine all factors with equal weights
    final_factor = (vol_scaled_mom + intraday_factor + pv_efficiency_factor + breakout_factor) / 4
    
    # Apply mild mean reversion for extreme values
    factor_zscore = (final_factor - final_factor.rolling(window=20).mean()) / final_factor.rolling(window=20).std()
    final_factor = np.where(factor_zscore > 2, final_factor * 0.8, 
                           np.where(factor_zscore < -2, final_factor * 0.8, final_factor))
    
    return final_factor
