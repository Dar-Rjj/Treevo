import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Relative Momentum and Microstructure Factor
    Combines momentum divergence, order flow analysis, and liquidity regime classification
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient history for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Cross-Asset Momentum Comparison
        # Stock vs Sector Momentum Divergence (using market proxy)
        if i >= 5:
            stock_5d_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1)
            market_5d_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1)  # Using same stock as proxy
            rel_momentum_5d = stock_5d_return / (market_5d_return + 1e-8)
        else:
            rel_momentum_5d = 0
            
        if i >= 10:
            stock_10d_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1)
            market_10d_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1)
            market_adj_momentum = stock_10d_return - market_10d_return
        else:
            market_adj_momentum = 0
            
        # Cross-Asset Momentum Persistence
        momentum_persistence = 0
        if i >= 5:
            recent_returns = []
            for j in range(1, 6):
                if i-j >= 0:
                    ret = (current_data['close'].iloc[i-j+1] / current_data['close'].iloc[i-j] - 1)
                    recent_returns.append(ret)
            if len(recent_returns) >= 3:
                momentum_persistence = np.mean(recent_returns[-3:]) / (np.std(recent_returns[-3:]) + 1e-8)
        
        # Order Flow Microstructure Analysis
        # Intraday Price Pressure
        if i >= 1:
            close_to_close = (current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1)
            intraday_high_low = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i]
            eod_momentum = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / current_data['open'].iloc[i]
        else:
            close_to_close = intraday_high_low = eod_momentum = 0
            
        # Volume-Weighted Price Efficiency
        if i >= 5:
            # Calculate VWAP for current day
            typical_price = (current_data['high'].iloc[i] + current_data['low'].iloc[i] + current_data['close'].iloc[i]) / 3
            vwap_current = typical_price  # Simplified VWAP for current day
            vwap_deviation = (current_data['close'].iloc[i] - vwap_current) / vwap_current
            
            # Volume concentration (current volume vs recent average)
            recent_volume_avg = current_data['volume'].iloc[i-5:i].mean()
            volume_concentration = current_data['volume'].iloc[i] / (recent_volume_avg + 1e-8)
        else:
            vwap_deviation = volume_concentration = 0
            
        # Bid-Ask Spread Implied Factors (using high-low range as proxy)
        daily_range = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i]
        if i >= 5:
            avg_range = (current_data['high'].iloc[i-5:i+1] - current_data['low'].iloc[i-5:i+1]).mean() / current_data['close'].iloc[i-5:i+1].mean()
            range_vs_avg = daily_range / (avg_range + 1e-8)
        else:
            range_vs_avg = 0
            
        # Volume per price tick movement
        if daily_range > 0:
            volume_per_tick = current_data['volume'].iloc[i] / (daily_range * current_data['close'].iloc[i] + 1e-8)
        else:
            volume_per_tick = 0
            
        # Liquidity Regime Classification
        # Volume Profile Analysis
        if i >= 20:
            volume_20d_median = current_data['volume'].iloc[i-20:i].median()
            volume_vs_median = current_data['volume'].iloc[i] / (volume_20d_median + 1e-8)
            
            # Volume stability (inverse of coefficient of variation)
            volume_std = current_data['volume'].iloc[i-20:i].std()
            volume_stability = volume_20d_median / (volume_std + 1e-8)
        else:
            volume_vs_median = volume_stability = 0
            
        # Market Depth Proxy
        if current_data['volume'].iloc[i] > 0:
            price_impact = daily_range / (current_data['volume'].iloc[i] + 1e-8)
            range_volume_ratio = daily_range / (current_data['volume'].iloc[i] / current_data['close'].iloc[i] + 1e-8)
        else:
            price_impact = range_volume_ratio = 0
            
        # Liquidity Transition Detection
        if i >= 3:
            volume_acceleration = (current_data['volume'].iloc[i] - current_data['volume'].iloc[i-2]) / (current_data['volume'].iloc[i-2] + 1e-8)
            if i >= 5:
                range_expansion = daily_range / ((current_data['high'].iloc[i-5:i] - current_data['low'].iloc[i-5:i]).mean() / current_data['close'].iloc[i-5:i].mean() + 1e-8)
            else:
                range_expansion = 0
        else:
            volume_acceleration = range_expansion = 0
            
        # Multi-Timeframe Signal Integration with Dynamic Weighting
        # Short-term signals (1-3 days)
        short_term_signal = 0.4 * eod_momentum + 0.3 * vwap_deviation + 0.3 * volume_concentration
        
        # Medium-term signals (5-10 days)
        medium_term_signal = 0.5 * rel_momentum_5d + 0.3 * market_adj_momentum + 0.2 * momentum_persistence
        
        # Long-term signals (15-20 days)
        long_term_signal = 0.4 * volume_stability + 0.3 * range_volume_ratio + 0.3 * volume_acceleration
        
        # Dynamic weighting based on recent volatility
        if i >= 10:
            recent_volatility = current_data['close'].iloc[i-10:i].pct_change().std()
            volatility_weight = 1 / (1 + recent_volatility)  # Lower weight in high volatility
        else:
            volatility_weight = 1
            
        # Combine signals with dynamic weights
        final_signal = volatility_weight * (
            0.4 * short_term_signal + 
            0.4 * medium_term_signal + 
            0.2 * long_term_signal
        )
        
        result.iloc[i] = final_signal
    
    return result
