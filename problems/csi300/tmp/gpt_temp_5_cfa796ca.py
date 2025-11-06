import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate required periods
    for i in range(len(data)):
        if i < 9:  # Need at least 9 periods for calculations
            alpha.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # Multi-Timeframe Fractal Momentum
        # Short-Term Fractal Dimension
        short_term_high_low = data.iloc[i]['high'] - data.iloc[i]['low']
        short_term_std = current_data.iloc[i-2:i+1]['close'].std()
        short_fractal = short_term_high_low / (short_term_std + 0.001)
        
        # Medium-Term Fractal Dimension
        medium_high = current_data.iloc[i-4:i+1]['high'].max()
        medium_low = current_data.iloc[i-4:i+1]['low'].min()
        medium_range = medium_high - medium_low
        medium_std = current_data.iloc[i-4:i+1]['close'].std()
        medium_fractal = medium_range / (medium_std + 0.001)
        
        # Fractal Regime Classification
        fractal_avg = current_data.iloc[i-4:i+1]['high'].std()  # Using high std as proxy for fractal avg
        if medium_fractal > 1.3 * fractal_avg:
            regime = "high"
        elif medium_fractal < 0.8 * fractal_avg:
            regime = "low"
        else:
            regime = "normal"
        
        # Price Momentum
        price_momentum = (data.iloc[i]['close'] / data.iloc[i-5]['close'] - 1) / (current_data.iloc[i-9:i+1]['close'].std() + 0.001)
        
        # Fractal-Adjusted Momentum
        fractal_adjusted_momentum = price_momentum * (1 + (medium_fractal / (fractal_avg + 0.001) - 1))
        
        # Volume-Amount Confirmation Framework
        # Volume Spike Indicator
        volume_avg = current_data.iloc[i-4:i+1]['volume'].mean()
        volume_spike = data.iloc[i]['volume'] / (volume_avg + 0.001)
        
        # Amount Momentum
        amount_avg = current_data.iloc[i-4:i+1]['amount'].mean()
        amount_momentum = (data.iloc[i]['amount'] / (amount_avg + 0.001)) - 1
        
        # Volume-Amount Correlation
        vol_amount_corr = current_data.iloc[i-4:i+1]['volume'].corr(current_data.iloc[i-4:i+1]['amount'])
        if pd.isna(vol_amount_corr):
            vol_amount_corr = 0
        
        # Price-Volume-Amount Efficiency
        pct_change = (data.iloc[i]['close'] / data.iloc[i-1]['close'] - 1)
        price_volume_efficiency = pct_change / (data.iloc[i]['volume'] * data.iloc[i]['amount'] + 0.001)
        
        # Confirmation Score
        confirmation_score = volume_spike * (1 + amount_momentum) * (1 + vol_amount_corr) * (1 + abs(price_volume_efficiency))
        
        # Intraday Fractal Efficiency Patterns
        # Capture Ratio
        daily_range = data.iloc[i]['high'] - data.iloc[i]['low']
        capture_ratio = (data.iloc[i]['close'] - data.iloc[i]['open']) / (daily_range + 0.001)
        
        # Gap Efficiency
        gap = abs(data.iloc[i]['open'] - data.iloc[i-1]['close'])
        gap_efficiency = gap / (daily_range + 0.001)
        
        # Fractal Range Utilization
        fractal_range_util = abs(data.iloc[i]['close'] - data.iloc[i]['open']) / (daily_range + 0.001)
        
        # Intraday Fractal Score
        intraday_fractal_score = (medium_fractal / (fractal_avg + 0.001)) * capture_ratio
        
        # Efficiency Composite
        efficiency_composite = (capture_ratio + fractal_range_util) / (1 + abs(gap_efficiency)) * intraday_fractal_score
        
        # Fractal Transition Dynamics
        # Fractal Regime Persistence (simplified)
        regime_persistence = 0.6  # Placeholder for regime persistence calculation
        
        # Recent Fractal Change
        if i >= 1:
            prev_fractal = (current_data.iloc[i-1]['high'] - current_data.iloc[i-1]['low']) / (current_data.iloc[i-1]['close'] + 0.001)
            recent_fractal_change = medium_fractal / (prev_fractal + 0.001) - 1
        else:
            recent_fractal_change = 0
        
        # Fractal Volatility
        fractal_volatility = current_data.iloc[i-4:i+1]['high'].std()  # Using high std as proxy
        
        # Transition Signal
        transition_signal = np.sign(recent_fractal_change) * (1 - regime_persistence)
        
        # Transition Quality
        transition_quality = abs(recent_fractal_change) / (1 + fractal_volatility)
        
        # Multi-Scale Divergence Detection
        # Very Short Momentum
        very_short_momentum = data.iloc[i]['close'] / data.iloc[i-1]['close'] - 1
        
        # Short Momentum
        short_momentum = data.iloc[i]['close'] / data.iloc[i-3]['close'] - 1
        
        # Medium Momentum
        medium_momentum = data.iloc[i]['close'] / data.iloc[i-8]['close'] - 1
        
        # Fractal-Momentum Alignment (simplified correlation)
        fractal_momentum_alignment = 0.5  # Placeholder for correlation calculation
        
        # Divergence Score
        divergence_score = np.sign(very_short_momentum) * np.sign(short_momentum) * np.sign(medium_momentum) * (1 + fractal_momentum_alignment)
        
        # Composite Alpha Generation
        # Core Fractal Momentum
        core_fractal_momentum = fractal_adjusted_momentum * (1 + transition_signal)
        
        # Volume-Amount Enhanced
        volume_amount_enhanced = core_fractal_momentum * confirmation_score
        
        # Efficiency Weighted
        efficiency_weighted = volume_amount_enhanced * efficiency_composite
        
        # Transition Quality Filter
        transition_quality_filter = efficiency_weighted * (1 + transition_quality)
        
        # Final Alpha
        if divergence_score > 0:
            divergence_multiplier = 1.5
        elif divergence_score < 0:
            divergence_multiplier = 0.5
        else:
            divergence_multiplier = 1.0
            
        final_alpha = transition_quality_filter * divergence_multiplier * (1 + regime_persistence)
        
        alpha.iloc[i] = final_alpha
    
    return alpha
