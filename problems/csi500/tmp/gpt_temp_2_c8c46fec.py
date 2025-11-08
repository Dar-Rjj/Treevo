import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multiple market microstructure insights.
    This factor captures liquidity momentum divergence patterns to predict mean reversion.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate typical price
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # 1. Calculate Liquidity Momentum
    # Compute volume-weighted price
    data['vw_price'] = data['volume'] * data['typical_price']
    
    # EMA of volume-weighted prices (8-day and 21-day)
    data['vw_price_ema8'] = data['vw_price'].ewm(span=8, adjust=False).mean()
    data['vw_price_ema21'] = data['vw_price'].ewm(span=21, adjust=False).mean()
    
    # Liquidity acceleration (ROC of volume-weighted prices)
    data['liquidity_roc8'] = (data['vw_price_ema8'] - data['vw_price_ema8'].shift(8)) / data['vw_price_ema8'].shift(8)
    data['liquidity_roc21'] = (data['vw_price_ema21'] - data['vw_price_ema21'].shift(21)) / data['vw_price_ema21'].shift(21)
    
    # Price momentum for comparison
    data['price_roc8'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    data['price_roc21'] = (data['close'] - data['close'].shift(21)) / data['close'].shift(21)
    
    # 2. Detect Divergence Patterns
    # Identify price-liquidity divergence
    data['divergence_bullish'] = ((data['price_roc8'] < 0) & (data['liquidity_roc8'] > 0)).astype(int)
    data['divergence_bearish'] = ((data['price_roc8'] > 0) & (data['liquidity_roc8'] < 0)).astype(int)
    
    # Calculate divergence magnitude
    data['divergence_magnitude'] = np.abs(data['price_roc8'] - data['liquidity_roc8'])
    
    # Track duration of divergence (rolling count of consecutive divergence days)
    data['divergence_duration'] = 0
    for i in range(1, len(data)):
        if data['divergence_bullish'].iloc[i] == 1 or data['divergence_bearish'].iloc[i] == 1:
            data['divergence_duration'].iloc[i] = data['divergence_duration'].iloc[i-1] + 1
    
    # 3. Generate Divergence Signal
    # Combine divergence factors
    data['raw_signal'] = data['divergence_magnitude'] * data['divergence_duration']
    
    # Adjust for market conditions using recent volatility
    returns = data['close'].pct_change()
    recent_volatility = returns.rolling(window=20).std()
    
    # Final factor: divergence signal adjusted by volatility
    data['factor'] = data['raw_signal'] / (recent_volatility + 1e-8)
    
    # Apply sign based on divergence type (negative for bearish, positive for bullish)
    data['factor'] = np.where(data['divergence_bearish'] == 1, -data['factor'], data['factor'])
    
    # Clean up and return
    factor_series = data['factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor_series
