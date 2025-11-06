import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Short-Term Reversal
    data['daily_return'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Calculate Volatility Measure (Average True Range over 10 days)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_10'] = data['tr'].rolling(window=10).mean()
    
    # Combine Reversal and Volatility
    data['vol_adj_reversal'] = data['daily_return'] / data['atr_10']
    data['vol_adj_reversal'] = data['vol_adj_reversal'] * np.sign(data['daily_return'])
    
    # Calculate Volume Momentum (linear regression slope over 5 days)
    def volume_slope(volumes):
        if len(volumes) < 5 or np.all(volumes == volumes.iloc[0]):
            return 0
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
        return slope / np.mean(volumes) if np.mean(volumes) != 0 else 0
    
    data['volume_momentum'] = data['volume'].rolling(window=5).apply(
        volume_slope, raw=False
    )
    
    # Calculate Amount Intensity
    data['amount_5d_avg'] = data['amount'].rolling(window=5).mean()
    data['amount_intensity'] = data['amount'].rolling(window=5).sum() / data['amount_5d_avg']
    data['amount_intensity'] = data['amount_intensity'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Combine Volume and Amount Components
    data['liquidity_accel'] = np.log(
        np.abs(data['volume_momentum'] * data['amount_intensity']) + 1e-8
    ) * np.sign(data['volume_momentum'] * data['amount_intensity'])
    
    # Integrate Price and Liquidity Signals
    data['price_liquidity_product'] = data['vol_adj_reversal'] * data['liquidity_accel']
    
    # Calculate rolling correlation between components over 10 days
    data['correlation_10d'] = data['vol_adj_reversal'].rolling(window=10).corr(data['liquidity_accel'])
    
    # Adjust final factor by correlation strength
    data['factor'] = data['price_liquidity_product'] * data['correlation_10d'].abs()
    data['factor'] = data['factor'] * np.sign(data['vol_adj_reversal'])
    
    return data['factor']
