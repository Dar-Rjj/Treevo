import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor combining multiple market microstructure insights
    """
    df = data.copy()
    
    # Intraday Momentum Persistence
    df['intraday_momentum'] = ((df['high'] + df['low']) / 2 - df['close'].shift(1)) / df['close'].shift(1)
    
    # Track momentum direction persistence
    df['momentum_sign'] = np.sign(df['intraday_momentum'])
    df['persistence_count'] = 0
    for i in range(1, len(df)):
        if df['momentum_sign'].iloc[i] == df['momentum_sign'].iloc[i-1]:
            df['persistence_count'].iloc[i] = df['persistence_count'].iloc[i-1] + 1
    
    df['momentum_persistence'] = df['intraday_momentum'] * df['persistence_count']
    df['volume_ratio_1'] = df['volume'] / df['volume'].shift(1)
    momentum_component = df['momentum_persistence'] * df['volume_ratio_1']
    
    # Volatility-Regime Adjusted Reversal
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['price_deviation'] = (df['close'] - df['ma_10']) / df['close'].shift(1)
    
    df['returns'] = df['close'].pct_change()
    df['vol_20'] = df['returns'].rolling(window=20).std()
    df['vol_60'] = df['returns'].rolling(window=60).std()
    df['vol_regime'] = df['vol_20'] / df['vol_60']
    
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_regime'] = df['volume'] / df['volume_ma_20']
    
    reversal_component = df['price_deviation'] * df['vol_regime'] * df['volume_regime']
    
    # Amount Flow Imbalance
    df['directional_amount'] = np.where(df['close'] > df['open'], df['amount'],
                                       np.where(df['close'] < df['open'], -df['amount'], 0))
    
    df['flow_5d'] = df['directional_amount'].rolling(window=5).sum()
    df['total_amount_5d'] = df['amount'].rolling(window=5).sum()
    df['flow_persistence'] = df['flow_5d'] / df['total_amount_5d']
    
    df['efficiency'] = df['amount'] / df['volume']
    flow_component = df['flow_persistence'] * df['efficiency']
    
    # Range Breakout with Volume Validation
    df['range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['ma_range_10'] = df['range'].rolling(window=10).mean()
    df['range_expansion'] = df['range'] / df['ma_range_10']
    
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio_2'] = df['volume'] / df['volume_ma_10']
    
    df['max_high_5'] = df['high'].rolling(window=5).max()
    df['breakout_strength'] = df['close'] / df['max_high_5']
    
    breakout_component = df['range_expansion'] * df['volume_ratio_2'] * df['breakout_strength']
    
    # Liquidity-Efficient Momentum
    df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
    
    df['liquidity_efficiency'] = df['amount'] / (df['volume'] * (df['high'] + df['low']) / 2)
    
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_trend'] = df['volume'] / df['volume_ma_5']
    
    liquidity_component = df['price_momentum'] * df['liquidity_efficiency'] * df['volume_trend']
    
    # Combine all components with equal weights
    alpha_factor = (momentum_component + reversal_component + flow_component + 
                   breakout_component + liquidity_component) / 5
    
    return alpha_factor
