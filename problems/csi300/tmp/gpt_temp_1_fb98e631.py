import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Adaptive Momentum
    # Calculate 10-day true range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_10'] = df['true_range'].rolling(window=10, min_periods=10).mean()
    
    # Calculate 20-day return volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20, min_periods=20).std()
    
    # Classify high/low volatility regime
    vol_median = df['volatility_20'].rolling(window=60, min_periods=60).median()
    high_vol_regime = df['volatility_20'] > vol_median
    
    # Calculate momentum components
    df['momentum_2'] = df['close'] / df['close'].shift(2) - 1
    df['momentum_15'] = df['close'] / df['close'].shift(15) - 1
    
    # Calculate momentum variance for high volatility regime
    df['momentum_var'] = df['momentum_2'].rolling(window=10, min_periods=10).var()
    
    # Adaptive momentum calculation
    adaptive_momentum = np.where(
        high_vol_regime,
        df['momentum_2'] / (1 + df['momentum_var']),
        df['momentum_15'] * (1 + df['momentum_15'].rolling(window=5, min_periods=5).mean())
    )
    
    # Intraday Range Efficiency Factor
    df['efficiency_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_trend'] = df['efficiency_ratio'].rolling(window=5, min_periods=5).mean()
    efficiency_change = df['efficiency_ratio'] - df['efficiency_trend']
    
    # Volume-Order Imbalance Detection
    df['large_trade_ratio'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['imbalance_5d'] = df['large_trade_ratio'].rolling(window=5, min_periods=5).mean()
    imbalance_intensity = df['large_trade_ratio'] - df['imbalance_5d']
    
    # Multi-Scale Price Fractality
    ranges_1d = df['high'] - df['low']
    ranges_3d = df['high'].rolling(window=3, min_periods=3).max() - df['low'].rolling(window=3, min_periods=3).min()
    ranges_5d = df['high'].rolling(window=5, min_periods=5).max() - df['low'].rolling(window=5, min_periods=5).min()
    
    # Hurst exponent approximation using rescaled range
    def hurst_approx(series, window=20):
        rs_series = []
        for i in range(len(series) - window + 1):
            window_data = series.iloc[i:i+window]
            mean_val = window_data.mean()
            cumulative_deviation = (window_data - mean_val).cumsum()
            r = cumulative_deviation.max() - cumulative_deviation.min()
            s = window_data.std()
            if s > 0:
                rs_series.append(r / s)
            else:
                rs_series.append(np.nan)
        return pd.Series(rs_series, index=series.index[window-1:])
    
    hurst_values = hurst_approx(df['close'], window=20)
    hurst_values = hurst_values.reindex(df.index)
    
    # Opening Auction Strength Indicator
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    opening_range_breakout = (df['high'] - df['open']) / df['open']
    
    # Volume-Volatility Confluence
    df['vol_vol_ratio'] = df['true_range'] / df['volume'].replace(0, np.nan)
    vol_vol_trend = df['vol_vol_ratio'].rolling(window=10, min_periods=10).mean()
    vol_vol_divergence = df['vol_vol_ratio'] - vol_vol_trend
    
    # Price-Momentum Asymmetry Factor
    positive_returns = df['returns'].clip(lower=0)
    negative_returns = abs(df['returns'].clip(upper=0))
    
    df['upside_momentum'] = positive_returns.rolling(window=10, min_periods=10).mean()
    df['downside_momentum'] = negative_returns.rolling(window=10, min_periods=10).mean()
    momentum_asymmetry = df['upside_momentum'] / (df['downside_momentum'] + 1e-8)
    
    # Tick-Size Efficiency
    df['effective_tick'] = df['amount'] / df['volume'].replace(0, np.nan)
    tick_variability = df['effective_tick'].rolling(window=10, min_periods=10).std()
    
    # Range-Expansion Probability
    range_expansion = ranges_1d / ranges_1d.rolling(window=10, min_periods=10).mean()
    range_volatility = ranges_1d.rolling(window=20, min_periods=20).std()
    
    # Momentum-Quality Score
    mom_3 = df['close'] / df['close'].shift(3) - 1
    mom_5 = df['close'] / df['close'].shift(5) - 1
    mom_8 = df['close'] / df['close'].shift(8) - 1
    
    momentum_consistency = (mom_3.rolling(window=5, min_periods=5).std() + 
                          mom_5.rolling(window=5, min_periods=5).std() + 
                          mom_8.rolling(window=5, min_periods=5).std()) / 3
    
    momentum_smoothness = 1 / (1 + momentum_consistency)
    
    # Combine all factors with appropriate weights
    factor = (
        0.25 * adaptive_momentum +
        0.15 * efficiency_change +
        0.12 * imbalance_intensity +
        0.10 * hurst_values.fillna(0) +
        0.08 * opening_range_breakout +
        0.08 * vol_vol_divergence +
        0.08 * momentum_asymmetry +
        0.07 * (1 / (1 + tick_variability)) +
        0.04 * range_expansion +
        0.03 * momentum_smoothness
    )
    
    # Clean up intermediate columns
    cols_to_drop = ['tr1', 'tr2', 'tr3', 'true_range', 'atr_10', 'returns', 
                   'volatility_20', 'momentum_2', 'momentum_15', 'momentum_var',
                   'efficiency_ratio', 'efficiency_trend', 'large_trade_ratio',
                   'imbalance_5d', 'opening_gap', 'vol_vol_ratio',
                   'upside_momentum', 'downside_momentum', 'effective_tick']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return pd.Series(factor, index=df.index, name='factor')
