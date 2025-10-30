import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volume-Price Fractal Efficiency
    # Calculate Efficiency Ratio
    net_movement = df['close'] - df['close'].shift(5)
    total_movement = (df['close'] - df['close'].shift(1)).abs().rolling(window=5).sum()
    efficiency_ratio = net_movement / total_movement.replace(0, np.nan)
    
    # Calculate Volume Trend
    def volume_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    volume_slope_vals = df['volume'].rolling(window=5).apply(volume_slope, raw=True)
    
    # Combine Components
    vp_factor = efficiency_ratio * volume_slope_vals
    
    # Liquidity-Adjusted Reversal
    # Identify Price Extremes
    close_rolling = df['close'].rolling(window=3, center=True)
    is_local_max = (df['close'] == close_rolling.max()) & (df['close'] > df['close'].shift(1)) & (df['close'] > df['close'].shift(-1))
    is_local_min = (df['close'] == close_rolling.min()) & (df['close'] < df['close'].shift(1)) & (df['close'] < df['close'].shift(-1))
    
    reversal_distance = np.zeros(len(df))
    reversal_distance[is_local_max] = (df['close'].shift(1) - df['close'])[is_local_max] / df['close'].shift(1)[is_local_max]
    reversal_distance[is_local_min] = (df['close'] - df['close'].shift(1))[is_local_min] / df['close'].shift(1)[is_local_min]
    
    # Assess Liquidity
    volume_range_ratio = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    volume_range_ma = volume_range_ratio.rolling(window=5).mean()
    liquidity_score = volume_range_ratio / volume_range_ma.replace(0, np.nan) - 1
    
    # Generate Signal
    reversal_signal = reversal_distance * liquidity_score
    reversal_signal[is_local_max] = -reversal_signal[is_local_max]
    
    # Volatility-Adaptive Momentum
    # Detect Volatility State
    returns = df['close'].pct_change()
    rolling_vol = returns.rolling(window=5).std()
    vol_median = rolling_vol.rolling(window=20).median()
    high_vol_state = (rolling_vol > vol_median).astype(int)
    
    # Calculate Momentum Signals
    short_momentum = df['close'].pct_change(2)
    medium_momentum = df['close'].pct_change(5)
    
    # Adaptive Combination
    vol_factor = high_vol_state * short_momentum + (1 - high_vol_state) * medium_momentum
    
    # Order Flow Pressure
    # Estimate Trade Direction
    daily_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / daily_range.replace(0, np.nan)
    pressure_signal = close_position * df['volume']
    
    # Assess Persistence
    pressure_sign = np.sign(pressure_signal)
    streak = pressure_sign.groupby((pressure_sign != pressure_sign.shift(1)).cumsum()).cumcount() + 1
    streak_multiplier = np.minimum(streak, 5)  # Cap at 5
    
    # Generate Final Signal
    order_flow_factor = pressure_signal * streak_multiplier
    
    # Combine all factors with equal weights
    final_factor = (vp_factor.fillna(0) + reversal_signal.fillna(0) + 
                   vol_factor.fillna(0) + order_flow_factor.fillna(0)) / 4
    
    return final_factor
