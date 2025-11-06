import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Persistence Factor with Volume-Volatility Alignment
    Combines momentum persistence patterns with volume-volatility regime analysis
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1]
        prev_prev_data = df.iloc[i-2]
        
        # 1. Intraday Momentum Persistence Component
        # Overnight return (previous close to current open)
        overnight_return = (current_data['open'] - prev_data['close']) / prev_data['close']
        
        # Intraday return (current open to current close)
        intraday_return = (current_data['close'] - current_data['open']) / current_data['open']
        
        # Momentum persistence score
        if overnight_return * intraday_return > 0:  # Same direction
            momentum_persistence = (abs(overnight_return) + abs(intraday_return)) * np.sign(overnight_return)
        else:  # Opposite direction
            momentum_persistence = (abs(intraday_return) - abs(overnight_return)) * np.sign(intraday_return)
        
        # 2. Volume-Volatility Regime Alignment Component
        # Short-term volatility (3-day high-low range normalized)
        vol_window = 3
        if i >= vol_window:
            recent_highs = [df.iloc[j]['high'] for j in range(i-vol_window+1, i+1)]
            recent_lows = [df.iloc[j]['low'] for j in range(i-vol_window+1, i+1)]
            volatility = (max(recent_highs) - min(recent_lows)) / prev_data['close']
        else:
            volatility = (current_data['high'] - current_data['low']) / current_data['close']
        
        # Volume relative to recent average
        vol_lookback = 5
        if i >= vol_lookback:
            avg_volume = np.mean([df.iloc[j]['volume'] for j in range(i-vol_lookback, i)])
            volume_ratio = current_data['volume'] / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        # Volume-volatility regime score
        if volatility > np.percentile([df.iloc[j]['high']/df.iloc[j]['low'] - 1 for j in range(max(0, i-10), i)], 70) if i > 10 else 0.02:
            # High volatility regime
            if volume_ratio > 1.2:
                regime_score = 1.0  # Informed trading
            else:
                regime_score = -0.5  # Speculative moves
        else:
            # Low volatility regime
            if volume_ratio > 1.2:
                regime_score = 0.5  # Accumulation/distribution
            else:
                regime_score = -1.0  # Market indifference
        
        # 3. Price-Level Memory Effect Component
        price_lookback = 10
        if i >= price_lookback:
            recent_prices = [df.iloc[j]['close'] for j in range(i-price_lookback, i)]
            current_price = current_data['close']
            
            # Count how many times current price level was visited recently
            price_tolerance = 0.005  # 0.5% tolerance
            visit_count = sum(1 for price in recent_prices if abs(price - current_price) / current_price <= price_tolerance)
            
            # Price attraction/repulsion score
            if visit_count >= 3:  # Frequently visited level
                # Check if price is rapidly departing
                price_change = (current_price - np.mean(recent_prices)) / np.mean(recent_prices)
                if abs(price_change) > 0.03:  # 3% departure
                    memory_score = np.sign(price_change) * 1.0  # Strong trend
                else:
                    memory_score = -0.5  # Equilibrium
            else:
                memory_score = 0.0  # Neutral
        else:
            memory_score = 0.0
        
        # 4. Combine components with weights
        final_factor = (
            0.4 * momentum_persistence +
            0.3 * regime_score +
            0.3 * memory_score
        )
        
        factor_values.iloc[i] = final_factor
    
    # Fill initial NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
