import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Parameters
    K = 20  # Volatility regime lookback
    L = 10  # Volume-price consistency lookback
    
    # Calculate rolling high-low range volatility
    daily_range = (df['high'] - df['low']) / df['close']
    rolling_volatility = daily_range.rolling(window=K).std()
    
    # Identify volatility regime relative to historical median
    vol_regime = (rolling_volatility > rolling_volatility.expanding().median()).astype(int)
    
    # Calculate intraday momentum
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate efficiency ratio
    price_change = abs(df['close'] - df['close'].shift(1))
    daily_range_abs = df['high'] - df['low']
    efficiency_ratio = price_change / daily_range_abs.replace(0, np.nan)
    
    # Calculate volume-weighted price trend
    volume_weighted_return = (df['close'].pct_change() * df['volume']).rolling(window=L).sum()
    total_volume = df['volume'].rolling(window=L).sum()
    vwap_trend = volume_weighted_return / total_volume.replace(0, np.nan)
    
    # Calculate trend persistence (short-term vs medium-term)
    short_term_trend = df['close'].pct_change(periods=3).rolling(window=5).mean()
    medium_term_trend = df['close'].pct_change(periods=10).rolling(window=10).mean()
    trend_persistence = np.sign(short_term_trend) * np.sign(medium_term_trend)
    
    # Volume-price consistency signal
    volume_price_consistency = vwap_trend * trend_persistence
    
    # Generate adaptive signal
    for i in range(len(df)):
        if pd.isna(vol_regime.iloc[i]) or pd.isna(efficiency_ratio.iloc[i]) or pd.isna(volume_price_consistency.iloc[i]):
            result.iloc[i] = np.nan
            continue
            
        if vol_regime.iloc[i] == 1:  # High volatility regime
            # Use efficiency ratio as mean reversion signal
            result.iloc[i] = -efficiency_ratio.iloc[i]
        else:  # Low volatility regime
            # Use volume-price consistency as momentum signal
            result.iloc[i] = volume_price_consistency.iloc[i]
    
    return result
