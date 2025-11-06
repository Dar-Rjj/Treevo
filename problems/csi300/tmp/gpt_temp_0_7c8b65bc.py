import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # Calculate ATR (Average True Range)
    data['prev_close'] = data['close'].shift(1)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Calculate short-term and medium-term volatility
    data['short_term_vol'] = data['tr'].rolling(window=5, min_periods=3).mean()
    data['medium_term_vol'] = data['tr'].rolling(window=20, min_periods=10).mean()
    data['vol_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Momentum Calculation with Volume Confirmation
    data['price_momentum_5d'] = (data['close'] / data['close'].shift(5) - 1)
    data['volume_momentum_5d'] = (data['volume'] / data['volume'].shift(5) - 1)
    
    # Volatility-Regime Adaptive Scaling
    # High volatility regime
    high_vol_mask = data['vol_ratio'] > 1.2
    # Low volatility regime
    low_vol_mask = data['vol_ratio'] < 0.8
    
    # Initialize base signal
    data['base_signal'] = data['price_momentum_5d'] * np.sign(data['volume_momentum_5d'])
    
    # Apply volatility regime adjustments
    data['vol_adjusted_signal'] = data['base_signal'].copy()
    
    # High volatility: scale by recent volatility
    data.loc[high_vol_mask, 'vol_adjusted_signal'] = (
        data.loc[high_vol_mask, 'base_signal'] * 
        (1 / (1 + data.loc[high_vol_mask, 'short_term_vol']))
    )
    
    # Low volatility: weight breakout signal by volume confirmation
    data.loc[low_vol_mask, 'vol_adjusted_signal'] = (
        data.loc[low_vol_mask, 'base_signal'] * 
        (1 + np.tanh(data.loc[low_vol_mask, 'volume_momentum_5d']))
    )
    
    # Intraday Pattern Enhancement
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Count consecutive same-direction days
    data['daily_return'] = data['close'].pct_change()
    data['direction'] = np.sign(data['daily_return'])
    data['consecutive_days'] = 0
    
    for i in range(1, len(data)):
        if data['direction'].iloc[i] == data['direction'].iloc[i-1] and not pd.isna(data['direction'].iloc[i-1]):
            data['consecutive_days'].iloc[i] = data['consecutive_days'].iloc[i-1] + 1
    
    # Multi-Timeframe Signal Integration
    # Short-term signal (3-5 day momentum with volume)
    data['short_term_momentum'] = data['close'].pct_change(periods=3)
    data['short_term_volume'] = data['volume'].pct_change(periods=3)
    data['short_term_signal'] = data['short_term_momentum'] * np.sign(data['short_term_volume'])
    
    # Medium-term signal (10-20 day volatility-adjusted)
    data['medium_term_momentum'] = data['close'].pct_change(periods=15)
    data['medium_term_volatility'] = data['close'].pct_change().rolling(window=15, min_periods=10).std()
    data['medium_term_signal'] = data['medium_term_momentum'] / (1 + data['medium_term_volatility'])
    
    # Final factor calculation
    data['factor'] = (
        0.4 * data['vol_adjusted_signal'] +
        0.3 * data['short_term_signal'] +
        0.2 * data['medium_term_signal'] +
        0.1 * data['intraday_strength'] *
        (1 + 0.1 * np.tanh(data['consecutive_days'] / 5))
    )
    
    return data['factor']
