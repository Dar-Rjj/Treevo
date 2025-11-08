import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining microstructure volume patterns and temporal return asymmetries
    """
    # Make copy to avoid modifying original dataframe
    data = df.copy()
    
    # Microstructure-Informed Volume Clustering
    # Calculate intraday volume concentration using open-close volume patterns
    data['intraday_volume_ratio'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    data['volume_clustering'] = data['intraday_volume_ratio'].rolling(window=5).std()
    
    # Compute volume persistence across multiple time horizons
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    data['volume_persistence'] = (data['volume_ma_5'] / data['volume_ma_10'] - 1).rolling(window=3).mean()
    
    # Identify abnormal volume clustering relative to historical patterns
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['abnormal_volume'] = data['volume_zscore'].abs()
    
    # Price-Level Dependent Volatility Patterns
    # Calculate volatility characteristics at different absolute price levels
    data['returns'] = data['close'].pct_change()
    data['volatility_5d'] = data['returns'].rolling(window=5).std()
    data['price_level'] = data['close'].rolling(window=10).mean()
    
    # Compute price-level specific volatility regimes
    data['price_vol_ratio'] = data['volatility_5d'] / data['price_level']
    data['vol_regime'] = (data['price_vol_ratio'] > data['price_vol_ratio'].rolling(window=20).median()).astype(int)
    
    # Temporal Asymmetry in Return Distributions
    # Calculate morning vs afternoon return patterns (using intraday proxies)
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['temporal_asymmetry'] = data['overnight_return'].rolling(window=5).std() - data['intraday_return'].rolling(window=5).std()
    
    # Compute day-of-week specific return characteristics
    if isinstance(data.index, pd.DatetimeIndex):
        data['day_of_week'] = data.index.dayofweek
        # Calculate average returns by day of week (using rolling window to avoid lookahead)
        dow_returns = []
        for i in range(len(data)):
            if i >= 20:  # Ensure sufficient history
                window_data = data.iloc[i-20:i]
                current_dow = data.iloc[i]['day_of_week']
                dow_mean_return = window_data[window_data['day_of_week'] == current_dow]['returns'].mean()
                dow_returns.append(dow_mean_return)
            else:
                dow_returns.append(np.nan)
        data['dow_expected_return'] = dow_returns
        data['dow_return_deviation'] = data['returns'] - data['dow_expected_return']
    
    # Order Flow Imbalance Persistence
    # Calculate directional volume flow using price movements as proxy
    data['price_trend'] = data['close'].rolling(window=3).mean() - data['close'].rolling(window=10).mean()
    data['volume_trend'] = data['volume'].rolling(window=3).mean() - data['volume'].rolling(window=10).mean()
    data['order_flow_imbalance'] = data['price_trend'] * data['volume_trend']
    
    # Compute order flow imbalance persistence
    data['ofi_persistence'] = data['order_flow_imbalance'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.all(x == x[0]) else 0
    )
    
    # Combine factors with appropriate weights
    # Volume clustering signal (positive when abnormal volume aligns with price movement)
    volume_signal = data['abnormal_volume'] * np.sign(data['returns']) * (1 - data['volume_clustering'].rank(pct=True))
    
    # Volatility regime signal (negative when high volatility at current price level)
    vol_signal = -data['vol_regime'] * data['price_vol_ratio']
    
    # Temporal asymmetry signal (positive when morning patterns favor returns)
    temporal_signal = data['temporal_asymmetry'] * np.sign(data['overnight_return'])
    
    # Order flow signal (positive when persistent order flow aligns with momentum)
    order_flow_signal = data['ofi_persistence'] * np.sign(data['order_flow_imbalance'])
    
    # Final alpha factor combining all components
    alpha = (
        0.4 * volume_signal.fillna(0) +
        0.3 * vol_signal.fillna(0) + 
        0.2 * temporal_signal.fillna(0) +
        0.1 * order_flow_signal.fillna(0)
    )
    
    return alpha
