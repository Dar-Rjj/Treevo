import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining momentum, reversal, volume interactions,
    trend analysis, and divergence patterns using only current and historical data.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate required technical indicators
    # True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Rolling calculations (using only past data)
    df['volatility_5d'] = df['tr'].rolling(window=5, min_periods=3).mean()
    df['volatility_20d'] = df['tr'].rolling(window=20, min_periods=10).mean()
    df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=3).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['close_ma_5'] = df['close'].rolling(window=5, min_periods=3).mean()
    df['close_ma_20'] = df['close'].rolling(window=20, min_periods=10).mean()
    
    # Calculate returns and momentum components
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['intraday_ret'] = df['close'] / df['open'] - 1
    
    # Volume change ratio
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    
    # Efficiency ratio (absolute return / path length)
    df['path_length'] = (abs(df['close'] - df['close'].shift(5)) + 
                        abs(df['close'].shift(1) - df['close'].shift(2)) +
                        abs(df['close'].shift(2) - df['close'].shift(3)) +
                        abs(df['close'].shift(3) - df['close'].shift(4)) +
                        abs(df['close'].shift(4) - df['close'].shift(5)))
    df['efficiency_ratio'] = abs(df['ret_5d']) / (df['path_length'] / df['close'].shift(5) + 1e-8)
    
    # Daily range
    df['daily_range'] = df['high'] - df['low']
    
    # Volume breakout indicator
    df['volume_breakout'] = (df['volume'] > df['volume_ma_20'] * 1.2).astype(int)
    
    # Moving average slope
    df['ma_slope'] = (df['close_ma_5'] - df['close_ma_5'].shift(3)) / df['close_ma_5'].shift(3)
    
    # Volume confirmation score
    df['volume_confirmation'] = np.where(
        df['volume'] > df['volume_ma_5'],
        np.minimum(df['volume'] / df['volume_ma_5'], 3),
        0.5
    )
    
    # Volatility regime classification
    df['vol_regime'] = np.where(
        df['volatility_5d'] > df['volatility_20d'] * 1.2,
        2,  # High volatility
        np.where(df['volatility_5d'] < df['volatility_20d'] * 0.8, 0, 1)  # Low volatility / Normal
    )
    
    # Price and volume oscillators
    df['price_oscillator'] = (df['close_ma_5'] - df['close_ma_20']) / df['close_ma_20']
    df['volume_oscillator'] = (df['volume_ma_5'] - df['volume_ma_20']) / df['volume_ma_20']
    
    # Multi-timeframe momentum
    df['momentum_short'] = df['ret_5d']
    df['momentum_medium'] = df['ret_10d']
    
    # Calculate composite alpha factor
    for i in range(len(df)):
        if i < 20:  # Skip initial period for reliable calculations
            alpha.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        
        # Component 1: Volatility-Adjusted Momentum
        vol_adj_momentum = current['ret_5d'] / (current['volatility_5d'] / current['close'] + 1e-8)
        
        # Component 2: Intraday Reversal with Volume Confirmation
        intraday_reversal = current['intraday_ret'] * current['volume_change']
        
        # Component 3: Volume-Scaled Range Breakout
        range_breakout = current['daily_range'] * current['volume_breakout']
        
        # Component 4: Efficiency-Weighted Price Change
        efficiency_weighted = current['ret_5d'] * current['efficiency_ratio']
        
        # Component 5: Volume-Confirmed Trend
        volume_confirmed_trend = current['ma_slope'] * current['volume_confirmation']
        
        # Component 6: Volatility-Regime Momentum
        vol_regime_multiplier = {0: 0.5, 1: 1.0, 2: 1.5}.get(current['vol_regime'], 1.0)
        regime_momentum = current['ret_10d'] * vol_regime_multiplier
        
        # Component 7: Volume-Price Divergence
        volume_price_divergence = current['price_oscillator'] - current['volume_oscillator']
        
        # Component 8: Multi-Timeframe Momentum
        momentum_alignment = np.sign(current['momentum_short']) * np.sign(current['momentum_medium'])
        
        # Combine components with weights
        composite_alpha = (
            0.15 * vol_adj_momentum +
            0.12 * intraday_reversal +
            0.13 * range_breakout +
            0.14 * efficiency_weighted +
            0.16 * volume_confirmed_trend +
            0.12 * regime_momentum +
            0.09 * volume_price_divergence +
            0.09 * momentum_alignment
        )
        
        alpha.iloc[i] = composite_alpha
    
    # Normalize the alpha factor
    alpha = (alpha - alpha.rolling(window=60, min_periods=20).mean()) / (alpha.rolling(window=60, min_periods=20).std() + 1e-8)
    
    return alpha
