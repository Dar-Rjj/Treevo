import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Weighted Liquidity Reversal Efficiency Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required components with proper shifting
    for i in range(5, len(data)):
        current_data = data.iloc[max(0, i-5):i+1]
        
        # Multi-Timeframe Reversal Detection
        # Intraday Reversal Patterns
        gap_reversal = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / max(current_data['high'].iloc[-1] - current_data['low'].iloc[-1], 1e-8)
        intraday_exhaustion = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / max(current_data['high'].iloc[-1] - current_data['low'].iloc[-1], 1e-8)
        
        failed_breakout = 0
        if (current_data['high'].iloc[-1] - current_data['high'].iloc[-2]) * (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) < 0:
            failed_breakout = 1
        
        # Range-Based Reversal
        range_price_reversal = (current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / max(current_data['high'].iloc[-1] - current_data['low'].iloc[-1], 1e-8)
        
        volatility_breakout = ((current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / 
                              max(current_data['high'].iloc[-2] - current_data['low'].iloc[-2], 1e-8)) * np.sign(current_data['close'].iloc[-1] - current_data['open'].iloc[-1])
        
        # Multi-Period Reversion
        recent_high = max(current_data['high'].iloc[-4:])
        recent_low = min(current_data['low'].iloc[-4:])
        short_term_overextension = (current_data['close'].iloc[-1] - current_data['close'].iloc[-4]) / max(recent_high - recent_low, 1e-8)
        
        price_clustering = 0
        for j in range(1, 6):
            if i - j >= 0:
                if abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-1-j]) / current_data['close'].iloc[-1] <= 0.005:
                    price_clustering += 1
        
        # Liquidity Flow Analysis
        # Volume Patterns
        volume_median = np.median(current_data['volume'].iloc[-5:-1])
        volume_spike = 1 if current_data['volume'].iloc[-1] > volume_median else 0
        
        volume_clustering = current_data['volume'].iloc[-1] / max(np.max(current_data['volume'].iloc[-6:-1]), 1e-8)
        
        # Liquidity Regimes
        dry_up = 0
        if (current_data['volume'].iloc[-1] < 0.7 * current_data['volume'].iloc[-2] and 
            (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) < (current_data['high'].iloc[-2] - current_data['low'].iloc[-2])):
            dry_up = 1
        
        flood_in = 0
        if (current_data['volume'].iloc[-1] > 1.5 * current_data['volume'].iloc[-2] and 
            current_data['volume'].iloc[-1] > current_data['volume'].iloc[-3]):
            flood_in = 1
        
        # Volume Distribution
        true_range = max(current_data['high'].iloc[-1] - current_data['low'].iloc[-1],
                        abs(current_data['high'].iloc[-1] - current_data['close'].iloc[-2]),
                        abs(current_data['low'].iloc[-1] - current_data['close'].iloc[-2]))
        
        volume_concentration = current_data['volume'].iloc[-1] / max(true_range, 1e-8)
        volume_skew = (current_data['volume'].iloc[-1] - volume_median) / max(current_data['high'].iloc[-1] - current_data['low'].iloc[-1], 1e-8)
        
        # Volatility Efficiency Framework
        movement_efficiency = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / max(current_data['high'].iloc[-1] - current_data['low'].iloc[-1], 1e-8)
        random_walk_deviation = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / max(current_data['high'].iloc[-1] - current_data['low'].iloc[-1], 1e-8)
        
        impact_ratio = movement_efficiency
        absorption = current_data['volume'].iloc[-1] / max(abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]), 1e-8)
        
        # Signal Integration
        # Volatility-Weighted Reversal
        range_reversal_volatility = range_price_reversal / max(true_range, 1e-8)
        failed_breakout_volatility = failed_breakout * ((current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / 
                                                       max(current_data['high'].iloc[-2] - current_data['low'].iloc[-2], 1e-8))
        
        # Liquidity-Confirmed Reversal
        volume_spike_reversal = range_price_reversal * (current_data['volume'].iloc[-1] / max(current_data['volume'].iloc[-6], 1e-8))
        dry_up_reversal = range_price_reversal * (current_data['volume'].iloc[-2] / max(current_data['volume'].iloc[-1], 1e-8))
        
        # Efficiency-Enhanced Reversal
        efficient_reversal = range_price_reversal * movement_efficiency
        absorption_reversal = range_price_reversal * absorption
        
        # Final Factor Calculation
        volatility_weighted_reversal = range_reversal_volatility + failed_breakout_volatility
        liquidity_confirmed_reversal = volume_spike_reversal + dry_up_reversal
        efficiency_enhanced_reversal = efficient_reversal + absorption_reversal
        
        core_signal = volatility_weighted_reversal * liquidity_confirmed_reversal
        efficiency_validation = core_signal * efficiency_enhanced_reversal
        
        # Final factor value
        result.iloc[i] = efficiency_validation
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
