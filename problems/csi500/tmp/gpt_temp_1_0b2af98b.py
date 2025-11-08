import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Momentum and Liquidity Spillover Alpha Factor
    Combines momentum signals with liquidity regime detection for improved return prediction
    """
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Required minimum data points for calculations
    min_periods = 20
    
    for i in range(len(df)):
        if i < min_periods:
            alpha.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # 1. Liquidity Regime Detection
        # Bid-Ask Spread Analysis (using volume and price range as proxy)
        recent_data = current_data.iloc[-10:]  # Last 10 periods
        avg_volume = recent_data['volume'].mean()
        price_range = (recent_data['high'] - recent_data['low']) / recent_data['close']
        avg_price_range = price_range.mean()
        
        # Spread volatility (using price range volatility as proxy)
        range_volatility = price_range.std()
        
        # Liquidity score: higher volume + lower price range volatility = better liquidity
        liquidity_score = (avg_volume / current_data['volume'].iloc[-20:].mean()) * \
                         (1 / (1 + range_volatility))
        
        # 2. Momentum Signals
        # Short-term momentum (5-day)
        mom_short = (current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1)
        
        # Medium-term momentum (10-day)
        mom_medium = (current_data['close'].iloc[-1] / current_data['close'].iloc[-11] - 1)
        
        # Volatility-adjusted momentum
        returns_10d = current_data['close'].iloc[-10:].pct_change().dropna()
        vol_10d = returns_10d.std()
        if vol_10d > 0:
            mom_vol_adj = mom_short / vol_10d
        else:
            mom_vol_adj = mom_short
        
        # 3. Cross-Timeframe Momentum Alignment
        # Check if short and medium momentum are aligned
        momentum_alignment = 1 if mom_short * mom_medium > 0 else -1
        
        # Momentum acceleration (change in momentum strength)
        mom_prev_short = (current_data['close'].iloc[-2] / current_data['close'].iloc[-7] - 1)
        mom_acceleration = mom_short - mom_prev_short
        
        # 4. Liquidity-Momentum Interaction
        # High liquidity regime: emphasize pure momentum
        if liquidity_score > 1.2:
            liquidity_weight = 1.0
            momentum_strength = mom_vol_adj
            
        # Low liquidity regime: apply conservative filters
        elif liquidity_score < 0.8:
            liquidity_weight = 0.3
            # Only use momentum if volatility-adjusted and aligned
            if abs(mom_vol_adj) > 0.02 and momentum_alignment > 0:
                momentum_strength = mom_vol_adj * 0.7
            else:
                momentum_strength = 0
                
        # Normal liquidity: balanced approach
        else:
            liquidity_weight = 0.7
            momentum_strength = mom_vol_adj * momentum_alignment
        
        # 5. Volume Confirmation
        volume_trend = 1 if current_data['volume'].iloc[-1] > current_data['volume'].iloc[-5:].mean() else 0.5
        
        # 6. Price Range Efficiency (narrow ranges during uptrends are positive)
        current_range = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / current_data['close'].iloc[-1]
        avg_range_5d = (current_data['high'].iloc[-5:] - current_data['low'].iloc[-5:]).div(current_data['close'].iloc[-5:]).mean()
        range_efficiency = 1 if (mom_short > 0 and current_range < avg_range_5d) else -1 if (mom_short < 0 and current_range > avg_range_5d) else 0
        
        # 7. Final Alpha Calculation
        base_signal = momentum_strength * liquidity_weight
        
        # Add volume confirmation
        volume_adjusted = base_signal * volume_trend
        
        # Add range efficiency adjustment
        range_adjusted = volume_adjusted * (1 + 0.2 * range_efficiency)
        
        # Scale by momentum acceleration for dynamic positioning
        acceleration_factor = 1 + 2 * mom_acceleration if abs(mom_acceleration) < 0.1 else 1
        
        final_alpha = range_adjusted * acceleration_factor
        
        # Apply bounds to prevent extreme values
        alpha.iloc[i] = np.clip(final_alpha, -0.1, 0.1)
    
    return alpha
