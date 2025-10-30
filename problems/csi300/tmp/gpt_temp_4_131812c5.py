import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Timeframe Intraday Momentum
    # Compute Intraday Strength Ratios
    data['intraday_strength_5d'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['intraday_strength_10d'] = data['intraday_strength_5d'].rolling(window=10, min_periods=1).mean()
    data['intraday_strength_20d'] = data['intraday_strength_5d'].rolling(window=20, min_periods=1).mean()
    
    # Calculate Momentum Divergence
    data['short_medium_div'] = (data['intraday_strength_5d'] - data['intraday_strength_10d']) / data['close']
    data['medium_long_div'] = (data['intraday_strength_10d'] - data['intraday_strength_20d']) / data['close']
    
    # Assess Volatility Environment
    # Calculate Historical Volatility
    data['daily_returns'] = data['close'].pct_change()
    data['hist_vol_20d'] = data['daily_returns'].rolling(window=20, min_periods=1).std()
    
    # Calculate Intraday Volatility
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    data['avg_intraday_vol_10d'] = data['intraday_vol'].rolling(window=10, min_periods=1).mean()
    
    # Calculate True Range and Volatility Regime
    prev_close = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - prev_close),
            abs(data['low'] - prev_close)
        )
    )
    data['avg_true_range_10d'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    data['volatility_regime'] = data['true_range'] > data['avg_true_range_10d']
    
    # Analyze Volume-Price Divergence
    # Calculate Price Deviation
    data['price_ma_10d'] = data['close'].rolling(window=10, min_periods=1).mean()
    data['price_deviation'] = (data['close'] - data['price_ma_10d']) / data['price_ma_10d']
    
    # Calculate Volume Deviation
    data['volume_ma_10d'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_deviation'] = (data['volume'] - data['volume_ma_10d']) / data['volume_ma_10d']
    
    # Compute Volume Confirmation
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ma_20d'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_trend_ratio'] = data['volume_ma_5d'] / data['volume_ma_20d']
    
    # Calculate Volume-Price Divergence Signal
    data['volume_price_div'] = data['price_deviation'] * data['volume_deviation']
    data['volume_price_div_5d_sum'] = data['volume_price_div'].rolling(window=5, min_periods=1).sum()
    
    # Generate Adaptive Alpha Signal
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['volatility_regime']):
            alpha_signal.iloc[i] = 0
            continue
            
        if data.iloc[i]['volatility_regime']:  # High Volatility Regime
            # Focus on Medium-Long Momentum Divergence with volatility scaling
            momentum_signal = data.iloc[i]['medium_long_div']
            
            # Combine historical and intraday volatility measures
            combined_vol = (data.iloc[i]['hist_vol_20d'] + data.iloc[i]['avg_intraday_vol_10d']) / 2
            if combined_vol > 0:
                momentum_signal = momentum_signal / combined_vol
            
            # Require strong volume confirmation
            volume_confirm = (data.iloc[i]['volume_trend_ratio'] > 1 and 
                            data.iloc[i]['volume_price_div_5d_sum'] > 0)
            
            alpha_signal.iloc[i] = momentum_signal if volume_confirm else 0
            
        else:  # Low Volatility Regime
            # Focus on Short-Medium Momentum Divergence
            momentum_signal = data.iloc[i]['short_medium_div']
            
            # Apply moderate volume filters
            volume_confirm = (data.iloc[i]['volume_trend_ratio'] > 0.8 and 
                            data.iloc[i]['volume_price_div_5d_sum'] > 0)
            
            alpha_signal.iloc[i] = momentum_signal if volume_confirm else 0
    
    return alpha_signal
