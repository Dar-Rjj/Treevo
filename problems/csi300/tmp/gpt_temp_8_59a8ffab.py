import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Price Path Complexity
    high_low_range = (data['high'] - data['low']) / data['open']
    abs_open_close = abs((data['close'] - data['open']) / data['open'])
    epsilon = 1e-8
    complexity = high_low_range / (abs_open_close + epsilon)
    complexity_volume_adjusted = complexity * np.log(1 + data['volume'])
    
    # Volatility Regime Breakout Signal
    returns = data['close'].pct_change()
    short_term_vol = returns.rolling(window=5).std()
    long_term_vol = returns.rolling(window=20).std()
    vol_breakout = (short_term_vol / long_term_vol) - 1
    price_momentum = data['close'].pct_change(5)
    vol_momentum_factor = vol_breakout * price_momentum
    
    # Volume-Price Divergence Factor
    price_ma = data['close'].rolling(window=10).mean()
    volume_ma = data['volume'].rolling(window=10).mean()
    
    def get_divergence_score(row):
        price_above = row['close'] > row['price_ma']
        volume_below = row['volume'] < row['volume_ma']
        price_below = row['close'] < row['price_ma']
        volume_above = row['volume'] > row['volume_ma']
        
        if price_above and volume_below:
            return -1  # bearish
        elif price_below and volume_above:
            return 1   # bullish
        else:
            return 0   # neutral
    
    divergence_df = pd.DataFrame({
        'close': data['close'],
        'volume': data['volume'],
        'price_ma': price_ma,
        'volume_ma': volume_ma
    })
    divergence_scores = divergence_df.apply(get_divergence_score, axis=1)
    divergence_factor = divergence_scores.rolling(window=5).sum() / 5
    
    # Opening Gap Persistence Factor
    morning_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    realized_move = abs(data['high'] - data['low']) / data['open']
    gap_persistence = abs(morning_gap / (realized_move + epsilon))
    gap_persistence_volume_weighted = gap_persistence * np.log(data['volume'])
    
    # Price Rejection at Extremes
    def get_rejection_score(row, prev_row):
        if pd.isna(prev_row['high']) or pd.isna(prev_row['low']):
            return 0
            
        upper_rejection = (row['high'] > prev_row['high'] and 
                          row['close'] < row['open'] and 
                          (row['high'] - row['close']) > (row['close'] - row['low']))
        
        lower_rejection = (row['low'] < prev_row['low'] and 
                          row['close'] > row['open'] and 
                          (row['close'] - row['low']) > (row['high'] - row['close']))
        
        if lower_rejection:
            return 1
        elif upper_rejection:
            return -1
        else:
            return 0
    
    rejection_scores = []
    for i in range(len(data)):
        if i == 0:
            rejection_scores.append(0)
        else:
            current_row = data.iloc[i]
            prev_row = data.iloc[i-1]
            rejection_scores.append(get_rejection_score(current_row, prev_row))
    
    rejection_scores = pd.Series(rejection_scores, index=data.index)
    cumulative_rejection = rejection_scores.rolling(window=3).sum() * data['volume']
    
    # Liquidity-Efficiency Ratio
    price_efficiency = abs(returns).rolling(window=10).std()
    volume_efficiency = data['volume'].rolling(window=10).std()
    efficiency_ratio = price_efficiency / (volume_efficiency + epsilon)
    trend_direction = np.sign(data['close'].pct_change(5))
    efficiency_factor = efficiency_ratio * trend_direction
    
    # Intraday Momentum Carryover
    morning_strength = (data['high'] - data['open']) / data['open']
    afternoon_strength = (data['close'] - data['high']) / data['high']
    
    def get_momentum_score(morning, afternoon):
        if morning > 0 and afternoon > 0:
            return 2   # strong uptrend
        elif morning < 0 and afternoon < 0:
            return -2  # strong downtrend
        elif morning > 0 and afternoon < 0:
            return 1   # weak uptrend
        elif morning < 0 and afternoon > 0:
            return -1  # weak downtrend
        else:
            return 0   # neutral
    
    momentum_scores = []
    for i in range(len(data)):
        momentum_scores.append(get_momentum_score(morning_strength.iloc[i], afternoon_strength.iloc[i]))
    
    momentum_scores = pd.Series(momentum_scores, index=data.index)
    trading_range = (data['high'] - data['low']) / data['open']
    momentum_factor = momentum_scores * trading_range
    
    # Volume-Weighted Price Acceleration
    price_acceleration = data['close'].pct_change(3) - data['close'].pct_change(1)
    volume_trend = data['volume'].pct_change(3)
    acceleration_volume = price_acceleration * volume_trend
    acceleration_factor = acceleration_volume.rolling(window=3).mean()
    
    # Combine all factors with equal weights
    final_factor = (
        complexity_volume_adjusted.fillna(0) +
        vol_momentum_factor.fillna(0) +
        divergence_factor.fillna(0) +
        gap_persistence_volume_weighted.fillna(0) +
        cumulative_rejection.fillna(0) +
        efficiency_factor.fillna(0) +
        momentum_factor.fillna(0) +
        acceleration_factor.fillna(0)
    ) / 8
    
    return final_factor
