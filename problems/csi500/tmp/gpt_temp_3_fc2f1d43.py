import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate returns for momentum computation
    data['returns'] = data['close'].pct_change()
    
    # 1. Calculate Momentum Divergence
    # Short-term momentum (3-day, 5-day returns)
    data['mom_3d'] = data['close'].pct_change(3)
    data['mom_5d'] = data['close'].pct_change(5)
    
    # Medium-term momentum (10-day, 15-day returns)
    data['mom_10d'] = data['close'].pct_change(10)
    data['mom_15d'] = data['close'].pct_change(15)
    
    # Calculate divergence as momentum differences
    data['mom_div_short'] = data['mom_3d'] - data['mom_5d']
    data['mom_div_medium'] = data['mom_10d'] - data['mom_15d']
    data['momentum_divergence'] = data['mom_div_short'] + data['mom_div_medium']
    
    # 2. Assess Volatility Regime
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 5-day average true range
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    
    # Classify volatility state using rolling percentiles
    data['volatility_percentile'] = data['atr_5d'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
    )
    
    # 3. Incorporate Liquidity Dynamics
    # Compute volume acceleration (3-day, 5-day volume slopes)
    def calculate_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window and not np.isnan(y).any():
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['volume_slope_3d'] = calculate_slope(data['volume'], 3)
    data['volume_slope_5d'] = calculate_slope(data['volume'], 5)
    data['volume_acceleration'] = data['volume_slope_3d'] + data['volume_slope_5d']
    
    # Detect liquidity regime shifts
    data['volume_ma_10d'] = data['volume'].rolling(window=10).mean()
    data['liquidity_regime'] = (data['volume'] > data['volume_ma_10d']).astype(int)
    
    # 4. Generate Breakout Signals
    # Measure momentum acceleration (change in momentum)
    data['momentum_acceleration'] = data['mom_3d'].diff(2)
    
    # Detect 20-day high/low breaks
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    
    data['breakout_high'] = ((data['close'] > data['high_20d'].shift(1)) & 
                            (data['close'] > data['open'])).astype(int)
    data['breakout_low'] = ((data['close'] < data['low_20d'].shift(1)) & 
                           (data['close'] < data['open'])).astype(int)
    
    # Weight breakout strength by acceleration
    data['breakout_strength'] = (data['breakout_high'] - data['breakout_low']) * data['momentum_acceleration']
    
    # 5. Combine Components
    # Multiply divergence by volume acceleration
    data['div_vol_component'] = data['momentum_divergence'] * data['volume_acceleration']
    
    # Scale by volatility regime (higher weight in moderate volatility)
    volatility_weight = 1 - abs(data['volatility_percentile'] - 0.5) * 2
    data['scaled_component'] = data['div_vol_component'] * volatility_weight
    
    # Apply breakout confirmation
    breakout_multiplier = 1 + abs(data['breakout_strength'])
    data['final_factor'] = data['scaled_component'] * breakout_multiplier
    
    # Generate directional probability weights
    # Normalize and apply smoothing
    factor = data['final_factor'].fillna(0)
    factor = factor.rolling(window=5, min_periods=1).mean()
    
    return factor
