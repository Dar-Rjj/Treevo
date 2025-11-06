import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    df['short_term_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['medium_term_momentum'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Divergence Component
    df['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
    
    # Price-volume correlation break
    rolling_corr = []
    for i in range(len(df)):
        if i >= 10:
            window = df.iloc[i-9:i+1]
            corr = window['close'].corr(window['volume'])
            rolling_corr.append(corr)
        else:
            rolling_corr.append(np.nan)
    df['price_volume_corr'] = rolling_corr
    df['corr_deviation'] = df['price_volume_corr'] - df['price_volume_corr'].rolling(window=20, min_periods=10).mean()
    
    # Volatility regime adjustment
    df['daily_returns'] = df['close'].pct_change()
    df['recent_volatility'] = df['daily_returns'].rolling(window=20, min_periods=10).std()
    volatility_median = df['recent_volatility'].rolling(window=60, min_periods=30).median()
    df['vol_regime'] = np.where(df['recent_volatility'] > volatility_median, 1, 0)  # 1 = high vol, 0 = low vol
    
    # Trend regime adjustment (ADX-like)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14, min_periods=7).mean()
    
    # +DM and -DM
    df['plus_dm'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0),
        0
    )
    df['minus_dm'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0),
        0
    )
    
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14, min_periods=7).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14, min_periods=7).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=14, min_periods=7).mean()
    
    # Trend classification
    df['trend_regime'] = np.where(
        (df['adx'] > 25) & (df['plus_di'] > df['minus_di']), 1,  # Strong uptrend
        np.where(
            (df['adx'] > 25) & (df['minus_di'] > df['plus_di']), -1,  # Strong downtrend
            0  # No clear trend
        )
    )
    
    # Dynamic combination with regime adjustments
    # Base momentum component
    momentum_component = 0.6 * df['short_term_momentum'] + 0.4 * df['medium_term_momentum']
    
    # Volume divergence component
    volume_component = 0.7 * df['volume_momentum'] + 0.3 * df['corr_deviation']
    
    # Regime-based weighting
    # In high volatility: emphasize momentum over volume
    # In strong trends: emphasize momentum direction
    momentum_weight = np.where(df['vol_regime'] == 1, 0.7, 0.5)
    volume_weight = 1 - momentum_weight
    
    # Adjust for trend strength
    trend_adjustment = np.where(
        df['trend_regime'] != 0,
        df['trend_regime'] * 0.2 * abs(momentum_component),
        0
    )
    
    # Final factor calculation
    factor = (momentum_weight * momentum_component + 
              volume_weight * volume_component + 
              trend_adjustment)
    
    return factor
