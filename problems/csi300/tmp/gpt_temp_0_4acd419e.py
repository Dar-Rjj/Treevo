import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Liquidity Fractal Dynamics factor combining:
    - Liquidity-adjusted price fractal dimension
    - Bid-ask flow imbalance persistence  
    - Fractal liquidity regime transitions
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required minimum data points
    min_periods = 20
    
    for i in range(len(df)):
        if i < min_periods:
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # 1. Liquidity-Adjusted Price Fractal
        # Volume-weighted fractal dimension: log(Amount)/log(High-Low)
        high_low_range = current_data['high'] - current_data['low']
        valid_range = high_low_range > 0
        fractal_dim = np.log(current_data['amount'].where(valid_range)) / np.log(high_low_range.where(valid_range))
        
        # Multi-timeframe liquidity clustering (5, 10, 20 days)
        fractal_5d = fractal_dim.rolling(window=5, min_periods=3).mean()
        fractal_10d = fractal_dim.rolling(window=10, min_periods=5).mean() 
        fractal_20d = fractal_dim.rolling(window=20, min_periods=10).mean()
        
        liquidity_clustering = (fractal_5d.iloc[-1] - fractal_10d.iloc[-1]) + (fractal_10d.iloc[-1] - fractal_20d.iloc[-1])
        
        # 2. Bid-Ask Flow Imbalance
        # Intraday flow direction persistence
        price_change = current_data['close'] - current_data['open']
        flow_direction = np.sign(price_change) * current_data['volume']
        flow_persistence = flow_direction.rolling(window=5, min_periods=3).std() / flow_direction.rolling(window=5, min_periods=3).mean()
        
        # Flow momentum divergence (large vs small trades proxy)
        large_trade_flow = (current_data['close'] - current_data['open']).rolling(window=3, min_periods=2).mean()
        small_trade_flow = (current_data['high'] - current_data['low']).rolling(window=3, min_periods=2).mean() / current_data['volume'].rolling(window=3, min_periods=2).mean()
        flow_divergence = large_trade_flow.iloc[-1] - small_trade_flow.iloc[-1]
        
        # 3. Fractal Liquidity Regime Transitions
        # Liquidity evaporation during high-low compression
        range_compression = (current_data['high'] - current_data['low']).rolling(window=10, min_periods=5).std()
        volume_trend = current_data['volume'].rolling(window=10, min_periods=5).mean()
        liquidity_evaporation = range_compression.iloc[-1] / (volume_trend.iloc[-1] + 1e-8)
        
        # Flow regeneration at fractal support/resistance
        recent_lows = current_data['low'].rolling(window=10, min_periods=5).min()
        recent_highs = current_data['high'].rolling(window=10, min_periods=5).max()
        current_close = current_data['close'].iloc[-1]
        
        support_distance = (current_close - recent_lows.iloc[-1]) / (recent_highs.iloc[-1] - recent_lows.iloc[-1] + 1e-8)
        resistance_distance = (recent_highs.iloc[-1] - current_close) / (recent_highs.iloc[-1] - recent_lows.iloc[-1] + 1e-8)
        
        flow_regeneration = np.where(
            support_distance < 0.1, 
            current_data['volume'].iloc[-1] / current_data['volume'].rolling(window=5, min_periods=3).mean().iloc[-1],
            np.where(
                resistance_distance < 0.1,
                -current_data['volume'].iloc[-1] / current_data['volume'].rolling(window=5, min_periods=3).mean().iloc[-1],
                0
            )
        )
        
        # Combine components with appropriate weights
        fractal_component = liquidity_clustering * 0.4
        flow_component = (flow_persistence.iloc[-1] + flow_divergence) * 0.3
        regime_component = (liquidity_evaporation + flow_regeneration) * 0.3
        
        result.iloc[i] = fractal_component + flow_component + regime_component
    
    # Normalize the final factor
    result = (result - result.rolling(window=min_periods, min_periods=min_periods).mean()) / (result.rolling(window=min_periods, min_periods=min_periods).std() + 1e-8)
    
    return result
