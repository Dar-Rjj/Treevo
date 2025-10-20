import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required components for each day
    for i in range(len(data)):
        if i < 20:  # Need at least 20 days for some calculations
            factor.iloc[i] = 0
            continue
            
        current_date = data.index[i]
        
        # 1. Intraday Momentum Acceleration
        # Current intraday momentum
        current_momentum = data['close'].iloc[i] - data['open'].iloc[i]
        
        # Previous intraday momentum
        if i >= 1:
            prev_momentum = data['close'].iloc[i-1] - data['open'].iloc[i-1]
        else:
            prev_momentum = 0
            
        momentum_acceleration = current_momentum - prev_momentum
        
        # Calculate True Range for current day
        tr1 = data['high'].iloc[i] - data['low'].iloc[i]
        if i >= 1:
            tr2 = abs(data['high'].iloc[i] - data['close'].iloc[i-1])
            tr3 = abs(data['low'].iloc[i] - data['close'].iloc[i-1])
            true_range = max(tr1, tr2, tr3)
        else:
            true_range = tr1
            
        # Calculate 5-day Average True Range
        atr_values = []
        for j in range(max(0, i-4), i+1):
            if j == 0:
                tr = data['high'].iloc[j] - data['low'].iloc[j]
            else:
                tr1_j = data['high'].iloc[j] - data['low'].iloc[j]
                tr2_j = abs(data['high'].iloc[j] - data['close'].iloc[j-1])
                tr3_j = abs(data['low'].iloc[j] - data['close'].iloc[j-1])
                tr = max(tr1_j, tr2_j, tr3_j)
            atr_values.append(tr)
        
        atr_5 = np.mean(atr_values) if atr_values else 1
        
        # Scale momentum acceleration by ATR
        if atr_5 != 0:
            momentum_component = momentum_acceleration / atr_5
        else:
            momentum_component = momentum_acceleration
        
        # 2. Volume-Adjusted Price Reversal
        # Calculate 3-day VWAP average
        vwap_values = []
        for j in range(max(0, i-2), i+1):
            typical_price = (data['high'].iloc[j] + data['low'].iloc[j] + data['close'].iloc[j]) / 3
            vwap = typical_price * data['volume'].iloc[j]
            vwap_values.append(vwap)
        
        avg_vwap = np.mean(vwap_values) if vwap_values else data['close'].iloc[i]
        
        # Price reversal component
        reversal_component = data['close'].iloc[i] - avg_vwap
        
        # Volume z-score (20-day lookback)
        volume_window = data['volume'].iloc[max(0, i-19):i+1]
        volume_mean = np.mean(volume_window)
        volume_std = np.std(volume_window)
        
        if volume_std != 0:
            volume_zscore = (data['volume'].iloc[i] - volume_mean) / volume_std
        else:
            volume_zscore = 0
            
        # Volume-adjusted reversal
        volume_adjusted_reversal = reversal_component * abs(volume_zscore)
        
        # 3. Bid-Ask Pressure Indicator
        # Effective spread proxy
        if (data['high'].iloc[i] + data['low'].iloc[i]) != 0:
            spread_proxy = (data['high'].iloc[i] - data['low'].iloc[i]) / ((data['high'].iloc[i] + data['low'].iloc[i]) / 2)
        else:
            spread_proxy = 0
            
        # Price-Volume Correlation (10-day)
        if i >= 9:
            close_window = data['close'].iloc[i-9:i+1]
            volume_window = data['volume'].iloc[i-9:i+1]
            
            if len(close_window) >= 2 and len(volume_window) >= 2:
                price_volume_corr = np.corrcoef(close_window, volume_window)[0, 1]
                if np.isnan(price_volume_corr):
                    price_volume_corr = 0
            else:
                price_volume_corr = 0
        else:
            price_volume_corr = 0
            
        # Bid-ask pressure
        bid_ask_pressure = spread_proxy * price_volume_corr
        
        # 4. Overnight Gap Persistence
        if i >= 1:
            overnight_return = (data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        else:
            overnight_return = 0
            
        # 5-day variance of overnight returns
        overnight_returns = []
        for j in range(max(1, i-4), i+1):
            if j >= 1:
                overnight_ret = (data['open'].iloc[j] - data['close'].iloc[j-1]) / data['close'].iloc[j-1]
                overnight_returns.append(overnight_ret)
        
        if len(overnight_returns) >= 2:
            overnight_variance = np.var(overnight_returns)
        else:
            overnight_variance = 1
            
        # Overnight gap persistence
        if overnight_variance != 0:
            gap_persistence = overnight_return / overnight_variance
        else:
            gap_persistence = overnight_return
        
        # Combine all components with equal weights
        factor_value = (
            momentum_component +
            volume_adjusted_reversal +
            bid_ask_pressure +
            gap_persistence
        )
        
        factor.iloc[i] = factor_value
    
    return factor
