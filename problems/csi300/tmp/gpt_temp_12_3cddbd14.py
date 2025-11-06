import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Momentum Divergence
    # Price Momentum: (Close_t - Close_{t-5}) / Close_{t-5}
    price_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Volume Momentum: (Volume_t - Volume_{t-5}) / Volume_{t-5}
    volume_momentum = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    
    # Divergence Signal: Price Momentum - Volume Momentum
    divergence_signal = price_momentum - volume_momentum
    
    # Volume Asymmetry
    # Create boolean masks for price movements
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    # Calculate rolling averages for upside and downside volume
    upside_volume = data['volume'].rolling(window=10).apply(
        lambda x: x[up_days.loc[x.index].fillna(False)].mean() if up_days.loc[x.index].fillna(False).any() else 0
    )
    
    downside_volume = data['volume'].rolling(window=10).apply(
        lambda x: x[down_days.loc[x.index].fillna(False)].mean() if down_days.loc[x.index].fillna(False).any() else 1
    )
    
    # Asymmetry Multiplier: Upside Volume / Downside Volume
    # Avoid division by zero
    asymmetry_multiplier = upside_volume / downside_volume.replace(0, 1)
    
    # Trading Efficiency
    # Price Impact: (High_t - Low_t) / (Amount_t / Volume_t)
    # Avoid division by zero in amount/volume
    avg_trade_size = data['amount'] / data['volume'].replace(0, 1)
    price_impact = (data['high'] - data['low']) / avg_trade_size.replace(0, 1)
    
    # Final Factor: Divergence Signal × Asymmetry Multiplier × Price Impact
    factor = divergence_signal * asymmetry_multiplier * price_impact
    
    return factor
