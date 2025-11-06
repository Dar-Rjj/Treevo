import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate typical price for better volume-weighted calculations
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate dollar volume for flow analysis
    dollar_volume = df['volume'] * typical_price
    
    # Order Flow Memory Asymmetry
    # Calculate buy/sell pressure using price movement and volume
    price_change = df['close'].pct_change()
    volume_change = df['volume'].pct_change()
    
    # Directional flow indicator (positive for buying pressure, negative for selling)
    flow_direction = np.sign(price_change) * np.sqrt(np.abs(price_change * volume_change))
    flow_direction = flow_direction.fillna(0)
    
    # Calculate autocorrelation differences for buy vs sell flows
    positive_flow = flow_direction.where(flow_direction > 0, 0)
    negative_flow = -flow_direction.where(flow_direction < 0, 0)
    
    # Rolling autocorrelations for buy and sell flows (5-day window)
    def rolling_autocorr(series, window=5):
        return series.rolling(window=window).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
        )
    
    buy_autocorr = rolling_autocorr(positive_flow, 5)
    sell_autocorr = rolling_autocorr(negative_flow, 5)
    flow_memory_asymmetry = buy_autocorr - sell_autocorr
    
    # Price Impact Asymmetry
    # Calculate immediate impact of trades
    high_low_range = (df['high'] - df['low']) / typical_price
    volume_weighted_range = high_low_range * df['volume'] / df['volume'].rolling(10).mean()
    
    # Separate impact for up vs down days
    up_day_impact = volume_weighted_range.where(price_change > 0, 0)
    down_day_impact = volume_weighted_range.where(price_change < 0, 0)
    
    # Rolling average impact ratios (10-day window)
    up_impact_ratio = up_day_impact.rolling(10).mean() / (up_day_impact.rolling(10).mean() + down_day_impact.rolling(10).mean())
    down_impact_ratio = down_day_impact.rolling(10).mean() / (up_day_impact.rolling(10).mean() + down_day_impact.rolling(10).mean())
    impact_asymmetry = up_impact_ratio - down_impact_ratio
    
    # Liquidity Restoration Asymmetry
    # Calculate price reversal patterns after large moves
    large_move_threshold = price_change.abs().rolling(20).quantile(0.7)
    large_up_moves = (price_change > large_move_threshold) & (price_change > 0)
    large_down_moves = (price_change < -large_move_threshold) & (price_change < 0)
    
    # Calculate reversal strength in subsequent periods (using only past data)
    def calculate_reversal(move_mask, lookforward=3):
        reversal = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            if move_mask.iloc[i] and i + lookforward < len(df):
                # Use only data up to current period for calculation
                current_price = df['close'].iloc[i]
                future_prices = df['close'].iloc[i+1:i+lookforward+1]
                if len(future_prices) > 0:
                    reversal.iloc[i] = (current_price - future_prices.mean()) / current_price
        return reversal
    
    up_reversal = calculate_reversal(large_up_moves, 3)
    down_reversal = -calculate_reversal(large_down_moves, 3)  # Negative for down moves
    
    # Rolling reversal asymmetry (15-day window)
    up_reversal_avg = up_reversal.rolling(15).mean()
    down_reversal_avg = down_reversal.rolling(15).mean()
    restoration_asymmetry = up_reversal_avg - down_reversal_avg
    
    # Composite Signal Generation
    # Normalize components
    flow_memory_norm = (flow_memory_asymmetry - flow_memory_asymmetry.rolling(20).mean()) / flow_memory_asymmetry.rolling(20).std()
    impact_norm = (impact_asymmetry - impact_asymmetry.rolling(20).mean()) / impact_asymmetry.rolling(20).std()
    restoration_norm = (restoration_asymmetry - restoration_asymmetry.rolling(20).mean()) / restoration_asymmetry.rolling(20).std()
    
    # Combine with weights based on recent volatility
    recent_volatility = price_change.rolling(10).std()
    vol_weight = 1 / (1 + recent_volatility)  # Lower weight in high volatility
    
    # Final composite signal
    composite_signal = (
        0.4 * flow_memory_norm + 
        0.35 * impact_norm + 
        0.25 * restoration_norm
    ) * vol_weight
    
    # Clean and return
    composite_signal = composite_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return composite_signal
