import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Convergence Factor
    Combines 20-day and 60-day price and volume momentum with alignment analysis
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate daily price changes
    price_changes = df['close'].diff()
    
    # Calculate daily volume changes
    volume_changes = df['volume'].diff()
    
    # Long-term Price Momentum (60-day)
    price_return_60d = (df['close'] / df['close'].shift(60)) - 1
    price_trend_strength_60d = price_changes.rolling(window=60).sum() / price_changes.abs().rolling(window=60).sum()
    
    # Medium-term Price Momentum (20-day)
    price_return_20d = (df['close'] / df['close'].shift(20)) - 1
    price_trend_strength_20d = price_changes.rolling(window=20).sum() / price_changes.abs().rolling(window=20).sum()
    
    # Long-term Volume Momentum (60-day)
    volume_return_60d = (df['volume'] / df['volume'].shift(60)) - 1
    volume_trend_strength_60d = volume_changes.rolling(window=60).sum() / volume_changes.abs().rolling(window=60).sum()
    
    # Medium-term Volume Momentum (20-day)
    volume_return_20d = (df['volume'] / df['volume'].shift(20)) - 1
    volume_trend_strength_20d = volume_changes.rolling(window=20).sum() / volume_changes.abs().rolling(window=20).sum()
    
    # Multi-Timeframe Alignment Analysis
    # Price momentum alignment
    price_alignment = (price_return_20d * price_return_60d > 0).astype(int)
    price_alignment_strength = price_return_20d * price_return_60d
    
    # Volume momentum alignment
    volume_alignment = (volume_return_20d * volume_return_60d > 0).astype(int)
    volume_alignment_strength = volume_return_20d * volume_return_60d
    
    # Cross-asset alignment within each timeframe
    cross_alignment_20d = (price_return_20d * volume_return_20d > 0).astype(int)
    cross_alignment_60d = (price_return_60d * volume_return_60d > 0).astype(int)
    
    # Calculate cross-asset alignment score
    cross_alignment_score = (cross_alignment_20d + cross_alignment_60d) / 2
    
    # Factor Integration
    for i in range(len(df)):
        if i < 60:  # Skip first 60 days due to rolling windows
            result.iloc[i] = 0
            continue
            
        # Collect all four components
        components = [
            price_return_20d.iloc[i],
            price_return_60d.iloc[i],
            volume_return_20d.iloc[i],
            volume_return_60d.iloc[i]
        ]
        
        # Check alignment status
        aligned_components = 0
        if price_alignment.iloc[i]:
            aligned_components += 1
        if volume_alignment.iloc[i]:
            aligned_components += 1
        if cross_alignment_20d.iloc[i]:
            aligned_components += 1
        if cross_alignment_60d.iloc[i]:
            aligned_components += 1
        
        # Base signal: average of all four return components
        base_signal = np.nanmean(components)
        
        # Apply confirmation multiplier based on alignment strength
        if aligned_components == 4:  # Strong Confirmation
            multiplier = 2.0
            signal_value = np.sqrt(np.abs(components[0] * components[1] * components[2] * components[3]))
            if np.sign(components[0]) == np.sign(components[1]) == np.sign(components[2]) == np.sign(components[3]):
                signal_value *= np.sign(components[0])
        elif aligned_components == 3:  # Medium Confirmation
            multiplier = 1.5
            aligned_values = [comp for j, comp in enumerate(components) 
                            if ((j < 2 and price_alignment.iloc[i]) or 
                                (j >= 2 and volume_alignment.iloc[i]) or
                                (j % 2 == 0 and cross_alignment_20d.iloc[i]) or
                                (j % 2 == 1 and cross_alignment_60d.iloc[i]))]
            signal_value = np.nanmean(aligned_values) if aligned_values else base_signal
        elif aligned_components == 2:  # Weak Confirmation
            multiplier = 1.2
            aligned_values = [comp for j, comp in enumerate(components) 
                            if ((j < 2 and price_alignment.iloc[i]) or 
                                (j >= 2 and volume_alignment.iloc[i]) or
                                (j % 2 == 0 and cross_alignment_20d.iloc[i]) or
                                (j % 2 == 1 and cross_alignment_60d.iloc[i]))]
            signal_value = np.nanmean(aligned_values) if aligned_values else base_signal
        else:  # No alignment
            multiplier = 0.8
            signal_value = base_signal
        
        # Incorporate trend strength measures as additional weights
        trend_strength_avg = np.nanmean([
            price_trend_strength_20d.iloc[i],
            price_trend_strength_60d.iloc[i],
            volume_trend_strength_20d.iloc[i],
            volume_trend_strength_60d.iloc[i]
        ])
        
        # Final factor calculation
        final_factor = signal_value * multiplier * (1 + abs(trend_strength_avg))
        
        result.iloc[i] = final_factor
    
    return result
