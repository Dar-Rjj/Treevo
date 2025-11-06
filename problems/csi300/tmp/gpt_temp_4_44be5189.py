import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining gap reversal probability, range expansion momentum,
    asymmetric volume pressure, momentum-volume clustering, and regime-dependent efficiency.
    """
    df = df.copy()
    
    # 1. Gap-Reversal Probability
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_direction'] = np.sign(df['overnight_gap'])
    df['gap_magnitude'] = np.abs(df['overnight_gap'])
    
    # Calculate historical gap reversal probability (5-day window)
    df['gap_reversal_prob'] = 0.0
    for i in range(5, len(df)):
        window = df.iloc[i-5:i]
        if len(window) > 0:
            gaps = window['overnight_gap'].dropna()
            if len(gaps) > 0:
                # Calculate probability that gap direction reverses intraday
                reversals = []
                for j in range(len(window)):
                    if j > 0:
                        gap_dir = np.sign(window['overnight_gap'].iloc[j])
                        intraday_return = (window['close'].iloc[j] - window['open'].iloc[j]) / window['open'].iloc[j]
                        if gap_dir * intraday_return < 0:  # Direction reversed
                            reversals.append(1)
                        else:
                            reversals.append(0)
                if len(reversals) > 0:
                    df.iloc[i, df.columns.get_loc('gap_reversal_prob')] = np.mean(reversals)
    
    gap_signal = -df['overnight_gap'] * df['gap_reversal_prob']
    
    # 2. Range-Expansion Momentum Quality
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_expansion'] = df['daily_range'] / df['daily_range'].rolling(window=5, min_periods=1).mean()
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_persistence'] = (df['momentum_5d'].rolling(window=3, min_periods=1).std() + 1e-8) / \
                                (np.abs(df['momentum_5d'].rolling(window=3, min_periods=1).mean()) + 1e-8)
    
    range_momentum_signal = df['range_expansion'] * df['momentum_5d'] / df['momentum_persistence']
    
    # 3. Asymmetric Volume Pressure
    df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_imbalance'] = (df['volume'] - df['volume_ma']) / df['volume_ma']
    df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Directional volume pressure
    df['up_volume_pressure'] = np.where(df['close'] > df['open'], df['volume_imbalance'], 0)
    df['down_volume_pressure'] = np.where(df['close'] < df['open'], -df['volume_imbalance'], 0)
    df['net_volume_pressure'] = df['up_volume_pressure'] + df['down_volume_pressure']
    
    volume_signal = df['net_volume_pressure'] * df['price_efficiency']
    
    # 4. Momentum-Volume Clustering
    df['momentum_cluster'] = df['momentum_5d'].rolling(window=5, min_periods=1).apply(
        lambda x: 1 if (x > 0).sum() >= 3 else (-1 if (x < 0).sum() >= 3 else 0), raw=True
    )
    df['volume_cluster'] = df['volume_imbalance'].rolling(window=5, min_periods=1).apply(
        lambda x: 1 if (x > 0).sum() >= 3 else (-1 if (x < 0).sum() >= 3 else 0), raw=True
    )
    
    cluster_signal = df['momentum_cluster'] * df['volume_cluster'] * df['momentum_5d']
    
    # 5. Regime-Dependent Efficiency
    df['volatility_10d'] = df['close'].pct_change().rolling(window=10, min_periods=1).std()
    df['efficiency_ratio'] = np.abs(df['close'] - df['close'].shift(5)) / \
                            (df['high'].rolling(window=5, min_periods=1).max() - 
                             df['low'].rolling(window=5, min_periods=1).min() + 1e-8)
    
    # Regime classification
    df['regime'] = 0
    high_vol = df['volatility_10d'] > df['volatility_10d'].rolling(window=20, min_periods=1).quantile(0.7)
    low_efficiency = df['efficiency_ratio'] < df['efficiency_ratio'].rolling(window=20, min_periods=1).quantile(0.3)
    
    df.loc[high_vol & low_efficiency, 'regime'] = 1  # Inefficient high-vol regime
    df.loc[~high_vol & ~low_efficiency, 'regime'] = 2  # Efficient low-vol regime
    
    # Regime-appropriate momentum
    regime_momentum = np.where(df['regime'] == 1, 
                              df['momentum_5d'] * df['volume_imbalance'],  # Volume-validated momentum in inefficient
                              df['momentum_5d'] * df['efficiency_ratio'])  # Efficiency-weighted in efficient
    
    # Combine all signals with weights
    alpha = (0.25 * gap_signal + 
             0.20 * range_momentum_signal + 
             0.25 * volume_signal + 
             0.15 * cluster_signal + 
             0.15 * regime_momentum)
    
    return pd.Series(alpha, index=df.index)
