import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volume-Price Efficiency Momentum
    # Volume-Weighted Price Trend
    df['volume_slope'] = df['volume'].rolling(window=5).apply(lambda x: linregress(range(5), x)[0] if len(x) == 5 else np.nan, raw=True)
    df['price_momentum'] = df['close'].pct_change(10).abs()
    
    # Range Efficiency Ratio
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['efficiency_ratio'] = abs(df['close'] - df['prev_close']) / df['true_range']
    
    # Volume Surge Adjustment
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['volatility_coef'] = df['true_range'].rolling(window=10).apply(
        lambda x: linregress(range(10), x)[0] if len(x) == 10 else np.nan, raw=True
    )
    
    # Pressure Reversal Dynamics
    df['pressure_index'] = (df['close'] - (df['high'] + df['low']) / 2) * df['volume']
    df['pressure_trend'] = df['pressure_index'].rolling(window=5).apply(
        lambda x: linregress(range(5), x)[0] if len(x) == 5 else np.nan, raw=True
    )
    
    # Liquidity-Enhanced Returns
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20)
    df['turnover_efficiency'] = df['amount'] / df['volume']
    
    # Gap Fill Probability
    df['overnight_gap'] = df['open'] / df['close'].shift(1) - 1
    df['gap_size'] = abs(df['overnight_gap'])
    df['volume_impact'] = df['volume'] / df['volume'].rolling(window=10).mean()
    
    # Volume Cluster Support
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume_cluster'] = df['volume'].rolling(window=5).mean()
    df['price_to_cluster'] = (df['close'] - df['vwap'].rolling(window=10).mean()) / df['close'].rolling(window=10).std()
    
    # Momentum Decay Pattern
    df['rolling_ret_5'] = df['close'].pct_change(5)
    df['decay_factor'] = df['rolling_ret_5'].ewm(span=5).mean()
    df['decay_acceleration'] = df['decay_factor'].diff()
    
    # Volume-Volatility Break
    df['range'] = df['high'] - df['low']
    df['vol_vol_corr'] = df['volume'].rolling(window=10).corr(df['range'])
    df['corr_regime'] = df['vol_vol_corr'].rolling(window=5).std()
    
    # Efficiency Trend Quality
    df['price_trend'] = df['close'].rolling(window=10).apply(
        lambda x: linregress(range(10), x)[0] if len(x) == 10 else np.nan, raw=True
    )
    df['price_noise'] = df['close'].rolling(window=10).std()
    df['trend_quality'] = df['price_trend'] / df['price_noise']
    df['volume_trend_persistence'] = df['trend_quality'] * df['volume_ratio']
    
    # Combine factors with appropriate weights
    factor = (
        0.15 * df['volume_slope'] * df['price_momentum'] +
        0.12 * df['efficiency_ratio'] * df['volume_ratio'] * df['volatility_coef'] +
        0.10 * df['pressure_index'] * df['pressure_trend'] +
        0.13 * (df['ret_5'] + 0.7 * df['ret_10'] + 0.5 * df['ret_20']) * df['turnover_efficiency'] +
        0.11 * (-df['overnight_gap'] * df['gap_size'] * df['volume_impact']) +
        0.09 * (-df['price_to_cluster'] * df['volume_cluster']) +
        0.10 * df['decay_factor'] * df['decay_acceleration'] +
        0.10 * df['vol_vol_corr'] * df['corr_regime'] +
        0.10 * df['trend_quality'] * df['volume_trend_persistence']
    )
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'true_range', 'volume_slope', 'price_momentum', 
                   'efficiency_ratio', 'volume_ratio', 'volatility_coef', 'pressure_index',
                   'pressure_trend', 'ret_5', 'ret_10', 'ret_20', 'turnover_efficiency',
                   'overnight_gap', 'gap_size', 'volume_impact', 'vwap', 'volume_cluster',
                   'price_to_cluster', 'rolling_ret_5', 'decay_factor', 'decay_acceleration',
                   'range', 'vol_vol_corr', 'corr_regime', 'price_trend', 'price_noise',
                   'trend_quality', 'volume_trend_persistence']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return factor
