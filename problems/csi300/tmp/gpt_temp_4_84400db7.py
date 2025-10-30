import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required technical indicators
    # ATR calculation
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_10 = true_range.rolling(window=10, min_periods=10).mean()
    atr_20 = true_range.rolling(window=20, min_periods=20).mean()
    
    # Volume statistics
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=5).mean()
    volume_ma_10 = data['volume'].rolling(window=10, min_periods=10).mean()
    volume_ma_20 = data['volume'].rolling(window=20, min_periods=20).mean()
    volume_std_20 = data['volume'].rolling(window=20, min_periods=20).std()
    
    # Price statistics
    close_ma_10 = data['close'].rolling(window=10, min_periods=10).mean()
    close_ma_20 = data['close'].rolling(window=20, min_periods=20).mean()
    
    # Calculate sub-factors
    for i in range(len(data)):
        if i < 20:  # Need sufficient history
            factor.iloc[i] = 0
            continue
            
        # 1. Momentum & Volume Acceleration
        if i >= 5:
            momentum_5d = (data['close'].iloc[i] / data['close'].iloc[i-5]) - 1
            # Volume trend slope (t-2 to t)
            if i >= 2:
                vol_slope = (data['volume'].iloc[i] - data['volume'].iloc[i-2]) / 2
                factor1 = momentum_5d * vol_slope
            else:
                factor1 = 0
        else:
            factor1 = 0
        
        # 2. Intraday Persistence
        intraday_strength = np.abs((data['high'].iloc[i] - data['low'].iloc[i]) / 
                                  (data['open'].iloc[i] - data['close'].iloc[i] + 1e-8))
        volume_ratio = data['volume'].iloc[i] / volume_ma_5.iloc[i]
        direction = np.sign(data['close'].iloc[i] - (data['high'].iloc[i] + data['low'].iloc[i])/2)
        factor2 = intraday_strength * volume_ratio * direction
        
        # 3. Liquidity-Adjusted Reversal
        ret_3d = (data['close'].iloc[i] / data['close'].iloc[i-3]) - 1
        volume_to_price = data['volume'].iloc[i] / data['close'].iloc[i]
        vol_price_ratio_ma_10 = (data['volume'] / data['close']).rolling(window=10, min_periods=10).median().iloc[i]
        factor3 = ret_3d * (volume_to_price / (vol_price_ratio_ma_10 + 1e-8))
        
        # 4. Volatility Breakout
        volatility_ratio = atr_10.iloc[i] / (atr_20.iloc[i] + 1e-8)
        volume_zscore = (data['volume'].iloc[i] - volume_ma_20.iloc[i]) / (volume_std_20.iloc[i] + 1e-8)
        direction_vol = np.sign(data['close'].iloc[i] - data['open'].iloc[i])
        factor4 = volatility_ratio * volume_zscore * direction_vol
        
        # 5. Price Efficiency
        price_noise = (data['high'].iloc[i] - data['low'].iloc[i]) / (np.abs(data['close'].iloc[i] - data['open'].iloc[i]) + 1e-8)
        dollar_volume = data['volume'].iloc[i] * data['close'].iloc[i]
        dollar_volume_ma_10 = (data['volume'] * data['close']).rolling(window=10, min_periods=10).mean().iloc[i]
        dollar_volume_ratio = dollar_volume / (dollar_volume_ma_10 + 1e-8)
        factor5 = price_noise / (dollar_volume_ratio + 1e-8)
        
        # 6. Order Flow Proxy
        pressure = (data['close'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i] + 1e-8)
        volume_adj_flow = pressure * data['volume'].iloc[i] / (volume_ma_10.iloc[i] + 1e-8)
        # 3-day cumulative
        if i >= 3:
            factor6 = sum([(data['close'].iloc[j] - data['low'].iloc[j]) / (data['high'].iloc[j] - data['low'].iloc[j] + 1e-8) * 
                          data['volume'].iloc[j] / (volume_ma_10.iloc[j] + 1e-8) for j in range(i-2, i+1)])
        else:
            factor6 = volume_adj_flow
        
        # 7. Volume Anomaly
        volume_surprise = data['volume'].iloc[i] / (volume_ma_20.iloc[i] + 1e-8)
        daily_ret = (data['close'].iloc[i] / data['close'].iloc[i-1]) - 1
        factor7 = volume_surprise * daily_ret
        
        # 8. Multi-Timeframe Momentum
        # Short-term: 3-day return / 10-day price range
        ret_3d_short = (data['close'].iloc[i] / data['close'].iloc[i-3]) - 1
        price_range_10d = data['high'].rolling(window=10, min_periods=10).max().iloc[i] - data['low'].rolling(window=10, min_periods=10).min().iloc[i]
        short_term = ret_3d_short / (price_range_10d + 1e-8)
        
        # Medium-term: 10-day return / 20-day price range
        ret_10d = (data['close'].iloc[i] / data['close'].iloc[i-10]) - 1
        price_range_20d = data['high'].rolling(window=20, min_periods=20).max().iloc[i] - data['low'].rolling(window=20, min_periods=20).min().iloc[i]
        medium_term = ret_10d / (price_range_20d + 1e-8)
        
        # Alignment sign
        alignment = np.sign(short_term * medium_term)
        factor8 = short_term * medium_term * alignment
        
        # Combine all factors with equal weights
        factors = [factor1, factor2, factor3, factor4, factor5, factor6, factor7, factor8]
        valid_factors = [f for f in factors if not (np.isnan(f) or np.isinf(f))]
        
        if valid_factors:
            factor.iloc[i] = np.mean(valid_factors)
        else:
            factor.iloc[i] = 0
    
    return factor
