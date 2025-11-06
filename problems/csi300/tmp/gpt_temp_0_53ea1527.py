import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required rolling windows
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['close_prev'] = data['close'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    
    # Calculate all components
    for i in range(len(data)):
        if i < 1:  # Skip first row due to lagged calculations
            factor.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # 1. Intraday Reversal with Volume
        intraday_return = (current['high'] - current['low']) / current['low']
        direction_signal = np.sign(current['close'] - current['open'])
        intraday_reversal = intraday_return * direction_signal * np.log(current['volume'] + 1)
        
        # 2. Volatility-Adjusted Turnover
        price_range = (current['high'] - current['low']) / ((current['high'] + current['low']) / 2)
        dollar_volume = current['volume'] * current['close']
        volatility_turnover = dollar_volume / (price_range + 1e-8)  # Avoid division by zero
        
        # 3. Abnormal Volume-Price Alignment
        volume_spike = current['volume'] / (current['volume_ma_20'] + 1e-8)
        price_direction = np.sign(current['close'] - prev['close'])
        volume_price_alignment = volume_spike * price_direction
        
        # 4. Momentum-Acceleration Convergence
        # Calculate rolling returns and volume growth
        if i >= 10:
            close_5d_ago = data.iloc[i-5]['close'] if i >= 5 else data.iloc[0]['close']
            close_10d_ago = data.iloc[i-10]['close']
            volume_5d_ago = data.iloc[i-5]['volume'] if i >= 5 else data.iloc[0]['volume']
            volume_10d_ago = data.iloc[i-10]['volume']
            
            price_momentum = (current['close'] - close_5d_ago) / close_5d_ago - (current['close'] - close_10d_ago) / close_10d_ago
            volume_acceleration = (current['volume'] - volume_5d_ago) / (volume_5d_ago + 1e-8) - (current['volume'] - volume_10d_ago) / (volume_10d_ago + 1e-8)
            momentum_convergence = price_momentum * volume_acceleration
        else:
            momentum_convergence = 0
        
        # 5. Liquidity Proxy Factor
        spread_proxy = (current['high'] - current['low']) / ((current['high'] + current['low']) / 2)
        volume_scale = np.log(current['volume'] + 1)
        liquidity_signal = -spread_proxy * volume_scale
        
        # 6. Gap-Momentum Consistency
        opening_gap = (current['open'] - prev['close']) / prev['close']
        intraday_momentum = (current['high'] - current['open']) / current['open']
        gap_persistence = opening_gap * intraday_momentum
        
        # 7. Volume-Ranked Price Efficiency
        efficiency_ratio = (current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        # Calculate rolling volume rank (using past 20 days)
        if i >= 20:
            recent_volumes = data['volume'].iloc[i-19:i+1]
            volume_rank = (current['volume'] > recent_volumes.median()).astype(int)
        else:
            volume_rank = 1
        volume_efficiency = efficiency_ratio * volume_rank
        
        # Combine all factors with equal weights
        combined_factor = (
            intraday_reversal + 
            volatility_turnover + 
            volume_price_alignment + 
            momentum_convergence + 
            liquidity_signal + 
            gap_persistence + 
            volume_efficiency
        )
        
        factor.iloc[i] = combined_factor
    
    # Normalize the factor
    factor = (factor - factor.mean()) / (factor.std() + 1e-8)
    
    return factor
