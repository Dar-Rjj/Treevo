import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Volatility-Adjusted Return Asymmetry
    df['skew_20d'] = df['returns'].rolling(window=20).skew()
    df['vol_20d'] = df['returns'].rolling(window=20).std()
    factor1 = df['skew_20d'] * df['vol_20d']
    
    # Momentum Acceleration Decay
    df['mom_5d'] = df['returns'].rolling(window=5).sum()
    df['mom_20d'] = df['returns'].rolling(window=20).sum()
    decay_weights = np.exp(-np.arange(20)/10)  # Exponential decay
    decay_weights = decay_weights / decay_weights.sum()
    df['mom_decay'] = df['returns'].rolling(window=20).apply(lambda x: np.sum(x * decay_weights), raw=True)
    factor2 = df['mom_5d'] - df['mom_decay']
    
    # Volume-Price Trend Divergence
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        return linregress(range(len(series)), series.values).slope
    
    df['price_slope_10d'] = df['close'].rolling(window=10).apply(calc_slope, raw=False)
    df['volume_slope_10d'] = df['volume'].rolling(window=10).apply(calc_slope, raw=False)
    factor3 = df['price_slope_10d'] * df['volume_slope_10d']
    
    # Liquidity-Weighted Reversal
    df['ret_1d'] = df['returns']
    df['ret_5d'] = df['returns'].rolling(window=5).sum()
    df['dollar_volume_5d'] = (df['close'] * df['volume']).rolling(window=5).mean()
    factor4 = (df['ret_1d'] - df['ret_5d']) * df['dollar_volume_5d']
    
    # Regime-Dependent Mean Reversion
    vol_median = df['vol_20d'].median()
    df['ma_20d'] = df['close'].rolling(window=20).mean()
    df['price_deviation'] = (df['close'] - df['ma_20d']) / df['ma_20d']
    high_vol_regime = (df['vol_20d'] > vol_median).astype(int)
    factor5 = df['price_deviation'] * (1 + high_vol_regime * 0.5)
    
    # Correlation Breakdown Signal (using rolling correlation with market proxy)
    df['market_returns'] = df['close'].pct_change()  # Using own returns as market proxy
    df['corr_20d'] = df['returns'].rolling(window=20).corr(df['market_returns'])
    df['corr_drop'] = df['corr_20d'].diff(5)  # 5-day change in correlation
    factor6 = df['corr_drop'] * df['returns'].rolling(window=5).sum()
    
    # Volume Asymmetry Ratio
    df['up_volume'] = df['volume'].where(df['returns'] > 0, 0).rolling(window=10).sum()
    df['down_volume'] = df['volume'].where(df['returns'] < 0, 0).rolling(window=10).sum()
    factor7 = np.log((df['up_volume'] + 1e-6) / (df['down_volume'] + 1e-6))
    
    # Intraday-Overnight Persistence
    df['intraday_ret'] = (df['close'] - df['open']) / df['open']
    df['overnight_ret'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['persistence'] = df['intraday_ret'] * df['overnight_ret']
    factor8 = df['persistence'].rolling(window=5).mean()
    
    # Tail Risk Probability Factor
    extreme_threshold = 2 * df['vol_20d']
    df['extreme_events'] = (df['returns'].abs() > extreme_threshold).rolling(window=60).sum()
    df['tail_prob'] = df['extreme_events'] / 60
    factor9 = df['tail_prob'] * df['vol_20d']
    
    # Multi-Horizon Momentum Consistency
    df['mom_5d_sign'] = np.sign(df['mom_5d'])
    df['mom_10d'] = df['returns'].rolling(window=10).sum()
    df['mom_10d_sign'] = np.sign(df['mom_10d'])
    df['mom_20d_sign'] = np.sign(df['mom_20d'])
    
    df['consistency_score'] = (df['mom_5d_sign'] + df['mom_10d_sign'] + df['mom_20d_sign']).abs() / 3
    strongest_momentum = df[['mom_5d', 'mom_10d', 'mom_20d']].abs().idxmax(axis=1)
    
    def get_strongest_mom(row):
        if pd.isna(strongest_momentum[row.name]):
            return np.nan
        return row[strongest_momentum[row.name]]
    
    df['strongest_mom'] = df.apply(get_strongest_mom, axis=1)
    factor10 = df['strongest_mom'] * df['consistency_score']
    
    # Combine all factors with equal weights
    factors = [factor1, factor2, factor3, factor4, factor5, 
               factor6, factor7, factor8, factor9, factor10]
    
    # Standardize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        standardized = (factor - factor.mean()) / factor.std()
        combined_factor += standardized
    
    return combined_factor
