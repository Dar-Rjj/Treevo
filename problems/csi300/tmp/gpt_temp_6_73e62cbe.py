import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Reversal with Volume-Price Divergence alpha factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Analysis
    # Short-Term Momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Medium-Term Momentum
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Compare short vs medium-term momentum directions
    data['momentum_dir_5_20'] = np.sign(data['momentum_5d']) * np.sign(data['momentum_20d'])
    
    # Volume Momentum Analysis
    # Volume Trend Calculation
    data['volume_change_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_avg_20d'] = data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Volume-Price Ratio Analysis
    data['volume_close_ratio'] = data['volume'] / data['close']
    data['volume_ratio_change_5d'] = data['volume_close_ratio'] / data['volume_close_ratio'].shift(5) - 1
    
    # Compare volume ratio trend with price trend
    data['volume_price_trend_align'] = np.sign(data['volume_ratio_change_5d']) * np.sign(data['momentum_5d'])
    
    # Reversal Detection
    # Identify Short-Term Reversals
    data['momentum_reversal_5_10'] = np.sign(data['momentum_5d']) * np.sign(data['momentum_10d'])
    
    # Check for oversold/overbought conditions using rolling percentiles
    data['momentum_5d_rank'] = data['momentum_5d'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Volume Confirmation of Reversals
    data['volume_spike'] = data['volume'] / data['volume_avg_20d'] - 1
    data['volume_trend_align'] = np.sign(data['volume_change_5d']) * np.sign(data['momentum_5d'])
    
    # Divergence Signal Generation
    # Price-Volume Divergence Analysis
    data['price_volume_divergence'] = np.sign(data['momentum_5d']) * np.sign(data['volume_change_5d'])
    
    # Identify bullish divergence (price down, volume up)
    bullish_divergence = (data['momentum_5d'] < 0) & (data['volume_change_5d'] > 0)
    # Identify bearish divergence (price up, volume down)
    bearish_divergence = (data['momentum_5d'] > 0) & (data['volume_change_5d'] < 0)
    
    # Combined Alpha Signal
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    # Strong signal when reversal confirmed by volume divergence
    strong_bullish = (
        (data['momentum_reversal_5_10'] < 0) &  # 5-day momentum reversing from 10-day
        (data['momentum_5d_rank'] < 0.2) &      # Oversold condition
        bullish_divergence &                     # Bullish divergence
        (data['volume_spike'] > 0.1)            # Volume spike confirmation
    )
    
    strong_bearish = (
        (data['momentum_reversal_5_10'] < 0) &  # 5-day momentum reversing from 10-day
        (data['momentum_5d_rank'] > 0.8) &      # Overbought condition
        bearish_divergence &                     # Bearish divergence
        (data['volume_spike'] > 0.1)            # Volume spike confirmation
    )
    
    # Moderate signal when only reversal or divergence present
    moderate_bullish = (
        ((data['momentum_reversal_5_10'] < 0) | bullish_divergence) &
        ~strong_bullish &
        (data['momentum_5d_rank'] < 0.3)
    )
    
    moderate_bearish = (
        ((data['momentum_reversal_5_10'] < 0) | bearish_divergence) &
        ~strong_bearish &
        (data['momentum_5d_rank'] > 0.7)
    )
    
    # Assign signal values
    alpha_signal[strong_bullish] = 2.0
    alpha_signal[strong_bearish] = -2.0
    alpha_signal[moderate_bullish] = 1.0
    alpha_signal[moderate_bearish] = -1.0
    
    # Weak signal for other momentum-volume alignment cases
    weak_bullish = (
        (data['price_volume_divergence'] < 0) & 
        (data['momentum_5d'] < 0) &
        ~strong_bullish & ~moderate_bullish
    )
    
    weak_bearish = (
        (data['price_volume_divergence'] < 0) & 
        (data['momentum_5d'] > 0) &
        ~strong_bearish & ~moderate_bearish
    )
    
    alpha_signal[weak_bullish] = 0.5
    alpha_signal[weak_bearish] = -0.5
    
    # Fill remaining with zero (no signal)
    alpha_signal = alpha_signal.fillna(0)
    
    return alpha_signal
