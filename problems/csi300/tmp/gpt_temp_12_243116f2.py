import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Volatility Regime Adjusted Momentum
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling volatility (20-day) and compare to historical median (252-day)
    data['vol_20d'] = data['true_range'].rolling(window=20).std()
    data['vol_median_252d'] = data['true_range'].rolling(window=252).std()
    data['vol_regime'] = data['vol_20d'] / data['vol_median_252d']
    
    # Price returns over multiple horizons
    data['ret_1d'] = data['close'].pct_change(1)
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_10d'] = data['close'].pct_change(10)
    
    # Scale momentum inversely with volatility
    data['mom_vol_adj'] = (data['ret_5d'] + data['ret_10d']) / (data['vol_regime'] + 1e-6)
    
    # Volume-Price Divergence Strength
    # Rolling linear regression slopes for price and volume (10-day window)
    def rolling_slope(series, window=10):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                x = np.arange(window)
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['price_slope'] = rolling_slope(data['close'], 10)
    data['volume_slope'] = rolling_slope(data['volume'], 10)
    
    # Correlation between trend directions (10-day rolling)
    data['price_dir'] = np.sign(data['price_slope'])
    data['volume_dir'] = np.sign(data['volume_slope'])
    data['dir_corr'] = data['price_dir'] * data['volume_dir']
    
    # Weight by trend magnitudes
    data['trend_strength'] = abs(data['price_slope'] * data['volume_slope'])
    data['volume_price_div'] = data['dir_corr'] * data['trend_strength']
    
    # Intraday Range Efficiency
    data['daily_range'] = data['high'] - data['low']
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['daily_range'] + 1e-6)
    
    # Volume percentile (20-day rolling)
    data['volume_percentile'] = data['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6), raw=False
    )
    data['range_efficiency'] = data['price_efficiency'] * data['volume_percentile']
    
    # Liquidity-Adjusted Reversal
    data['price_change'] = data['close'] - data['prev_close']
    data['liquidity'] = data['volume'] * abs(data['price_change'])
    data['liquidity_percentile'] = data['liquidity'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6), raw=False
    )
    data['reversal'] = -data['ret_1d'] * (1 - data['liquidity_percentile'])
    
    # Opening Gap Persistence
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_filled'] = ((data['opening_gap'] > 0) & (data['low'] <= data['prev_close'])) | \
                        ((data['opening_gap'] < 0) & (data['high'] >= data['prev_close']))
    data['gap_persistence'] = data['opening_gap'] * (~data['gap_filled']).astype(int)
    
    # Volume-Weighted Breakout
    # Local highs and lows (20-day window)
    data['local_high'] = data['high'].rolling(window=20, center=False).max()
    data['local_low'] = data['low'].rolling(window=20, center=False).min()
    
    # Breakout strength
    data['breakout_high'] = (data['close'] - data['local_high']) / data['local_high']
    data['breakout_low'] = (data['close'] - data['local_low']) / data['local_low']
    data['breakout_strength'] = np.where(
        data['breakout_high'] > 0, data['breakout_high'],
        np.where(data['breakout_low'] < 0, data['breakout_low'], 0)
    )
    data['volume_multiplier'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['volume_breakout'] = data['breakout_strength'] * data['volume_multiplier']
    
    # Price-Volume Acceleration
    data['price_accel'] = data['ret_1d'].diff()
    data['volume_ret'] = data['volume'].pct_change()
    data['volume_accel'] = data['volume_ret'].diff()
    data['accel_product'] = data['price_accel'] * data['volume_accel']
    
    # Intraday Volatility Clustering
    data['high_low_range'] = data['high'] - data['low']
    data['close_open_range'] = abs(data['close'] - data['open'])
    data['volatility_ratio'] = data['high_low_range'] / (data['close_open_range'] + 1e-6)
    
    # Regime-based prediction
    data['vol_cluster'] = np.where(
        data['volatility_ratio'] > data['volatility_ratio'].rolling(window=20).median(),
        -data['ret_1d'],  # High volatility → reversal
        data['ret_1d']    # Low volatility → trend
    )
    
    # Smart Money Flow
    data['amount_per_share'] = data['amount'] / (data['volume'] + 1e-6)
    data['large_txn'] = (data['amount_per_share'] > data['amount_per_share'].rolling(window=20).quantile(0.8)).astype(int)
    data['consecutive_large'] = data['large_txn'].rolling(window=3).sum()
    data['smart_money'] = data['consecutive_large'] * np.sign(data['price_change'])
    
    # Momentum-Volume Convergence
    data['price_mom_5d'] = data['close'].pct_change(5)
    data['price_mom_10d'] = data['close'].pct_change(10)
    data['volume_mom_5d'] = data['volume'].pct_change(5)
    data['volume_mom_10d'] = data['volume'].pct_change(10)
    
    data['mom_conv_dir'] = (np.sign(data['price_mom_5d']) == np.sign(data['volume_mom_5d'])).astype(int) + \
                          (np.sign(data['price_mom_10d']) == np.sign(data['volume_mom_10d'])).astype(int)
    data['mom_conv_strength'] = (abs(data['price_mom_5d']) + abs(data['price_mom_10d'])) * \
                               (abs(data['volume_mom_5d']) + abs(data['volume_mom_10d']))
    data['momentum_volume_conv'] = data['mom_conv_dir'] * data['mom_conv_strength']
    
    # Combine all factors with equal weights
    factors = [
        'mom_vol_adj', 'volume_price_div', 'range_efficiency', 'reversal',
        'gap_persistence', 'volume_breakout', 'accel_product', 'vol_cluster',
        'smart_money', 'momentum_volume_conv'
    ]
    
    # Standardize each factor and combine
    alpha = pd.Series(0, index=data.index)
    for factor in factors:
        if factor in data.columns:
            # Remove outliers and standardize
            factor_data = data[factor].replace([np.inf, -np.inf], np.nan)
            factor_data = (factor_data - factor_data.rolling(window=252).mean()) / (factor_data.rolling(window=252).std() + 1e-6)
            alpha += factor_data.fillna(0)
    
    # Clean up and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha
