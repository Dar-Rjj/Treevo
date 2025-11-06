import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Elastic Liquidity & Flow Asymmetry Alpha Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Extract current day data
        current = current_data.iloc[-1]
        open_price = current['open']
        high = current['high']
        low = current['low']
        close = current['close']
        amount = current['amount']
        volume = current['volume']
        
        # 1. Multi-Timeframe Elastic Momentum Analysis
        # Short-term momentum (1-day, 3-day)
        if i >= 3:
            mom_1d = (close - current_data.iloc[-2]['close']) / current_data.iloc[-2]['close']
            mom_3d = (close - current_data.iloc[-4]['close']) / current_data.iloc[-4]['close']
            mom_acceleration = mom_3d - mom_1d
            
            # Range Efficiency Ratio
            range_eff = (close - open_price) / (high - low) if (high - low) > 0 else 0
        else:
            mom_acceleration = 0
            range_eff = 0
        
        # Medium-term momentum (5-day, 10-day)
        if i >= 10:
            mom_5d = (close - current_data.iloc[-6]['close']) / current_data.iloc[-6]['close']
            mom_10d = (close - current_data.iloc[-11]['close']) / current_data.iloc[-11]['close']
            mom_stability = mom_10d / mom_5d if abs(mom_5d) > 1e-6 else 0
        else:
            mom_stability = 0
        
        # 2. Liquidity Regime Identification
        # Effective Spread Proxy at different intervals
        spread_t = (high - low) / close if close > 0 else 0
        
        if i >= 5:
            spread_t4 = (current_data.iloc[-5]['high'] - current_data.iloc[-5]['low']) / current_data.iloc[-5]['close']
        else:
            spread_t4 = spread_t
            
        if i >= 10:
            spread_t9 = (current_data.iloc[-10]['high'] - current_data.iloc[-10]['low']) / current_data.iloc[-10]['close']
        else:
            spread_t9 = spread_t
            
        if i >= 20:
            spread_t19 = (current_data.iloc[-20]['high'] - current_data.iloc[-20]['low']) / current_data.iloc[-20]['close']
        else:
            spread_t19 = spread_t
        
        # Liquidity regime classification
        spread_volatility = np.std([spread_t, spread_t4, spread_t9, spread_t19])
        avg_spread = np.mean([spread_t, spread_t4, spread_t9, spread_t19])
        liquidity_regime = 1 if avg_spread < np.percentile([spread_t, spread_t4, spread_t9, spread_t19], 33) else (
                           -1 if avg_spread > np.percentile([spread_t, spread_t4, spread_t9, spread_t19], 66) else 0)
        
        # 3. Flow Asymmetry & Smart Money Integration
        # Flow Bias
        flow_bias = (close - open_price) * amount / volume if volume > 0 else 0
        
        # Volume Efficiency
        volume_eff = volume / (high - low) if (high - low) > 0 else 0
        
        # Flow Concentration
        flow_concentration = amount / (volume * (high - low)) if (volume * (high - low)) > 0 else 0
        
        # 4. Range Compression Analysis
        if i >= 10:
            recent_ranges = [current_data.iloc[j]['high'] - current_data.iloc[j]['low'] for j in range(-10, 0)]
            avg_range = np.mean(recent_ranges)
            compression_intensity = (high - low) / avg_range if avg_range > 0 else 1
        else:
            compression_intensity = 1
        
        # 5. Composite Elastic Asymmetry Alpha
        # Momentum-flow convergence score
        momentum_flow_score = 0
        
        if i >= 3:
            # Short-term momentum-flow alignment
            st_momentum_flow = mom_1d * flow_bias * range_eff
            
            # Medium-term momentum stability with flow
            if i >= 10:
                mt_momentum_flow = mom_stability * flow_concentration * (1 - compression_intensity)
            else:
                mt_momentum_flow = 0
            
            # Cross-timeframe divergence
            cross_divergence = np.sign(mom_1d) * np.sign(mom_stability) if i >= 10 else 1
            
            momentum_flow_score = (st_momentum_flow * 0.6 + mt_momentum_flow * 0.4) * cross_divergence
        
        # 6. Liquidity regime weighting
        liquidity_weight = 1.0
        if liquidity_regime == 1:  # High liquidity
            liquidity_weight = 1.2
        elif liquidity_regime == -1:  # Low liquidity
            liquidity_weight = 0.8
        
        # 7. Volume absorption and smart money confirmation
        volume_absorption = volume_eff * flow_bias * (1 if flow_concentration > np.percentile([flow_concentration], 50) else 0.5)
        
        # 8. Final composite alpha
        elastic_asymmetry_alpha = (
            momentum_flow_score * 0.4 +
            volume_absorption * 0.3 +
            (mom_acceleration * range_eff) * 0.2 +
            (flow_concentration * compression_intensity) * 0.1
        ) * liquidity_weight
        
        result.iloc[i] = elastic_asymmetry_alpha
    
    return result
