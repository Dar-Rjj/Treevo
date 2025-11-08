import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Multi-Timeframe Persistence alpha factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Multi-timeframe periods
    timeframes = [5, 10, 20]
    timeframe_weights = {5: 0.3, 10: 0.4, 20: 0.3}
    
    # Store divergence signals for each timeframe
    divergence_signals = {}
    persistence_trackers = {}
    
    for tf in timeframes:
        # Price momentum calculations
        price_momentum = data['close'] / data['close'].shift(tf) - 1
        
        # EMA price momentum
        ema_short = data['close'].ewm(span=8).mean()
        ema_medium = data['close'].ewm(span=21).mean()
        ema_momentum = ema_short / ema_short.shift(5) - 1
        
        # Volume momentum calculations
        volume_momentum = data['volume'] / data['volume'].shift(tf) - 1
        
        # EMA volume momentum
        ema_vol_short = data['volume'].ewm(span=8).mean()
        ema_vol_medium = data['volume'].ewm(span=21).mean()
        ema_vol_momentum = ema_vol_short / ema_vol_short.shift(5) - 1
        
        # Combined momentum (simple average of raw and EMA)
        combined_price_momentum = (price_momentum + ema_momentum) / 2
        combined_volume_momentum = (volume_momentum + ema_vol_momentum) / 2
        
        # Divergence detection
        bullish_divergence = (combined_price_momentum < 0) & (combined_volume_momentum > 0)
        bearish_divergence = (combined_price_momentum > 0) & (combined_volume_momentum < 0)
        
        # Divergence magnitude scoring
        divergence_strength = abs(combined_price_momentum) * abs(combined_volume_momentum)
        
        # Initialize divergence signal
        divergence_signal = pd.Series(0.0, index=data.index)
        
        # Apply bullish divergence signals
        divergence_signal[bullish_divergence] = 1.0 * divergence_strength[bullish_divergence]
        
        # Apply bearish divergence signals
        divergence_signal[bearish_divergence] = -1.0 * divergence_strength[bearish_divergence]
        
        # Store for later aggregation
        divergence_signals[tf] = divergence_signal
        
        # Initialize persistence tracker for this timeframe
        persistence_trackers[tf] = pd.Series(0, index=data.index)
    
    # Calculate multi-timeframe divergence consistency
    consistency_scores = pd.Series(0.0, index=data.index)
    
    for i in range(len(data)):
        if i < max(timeframes):
            continue
            
        current_div_types = {}
        for tf in timeframes:
            signal = divergence_signals[tf].iloc[i]
            if signal > 0:
                current_div_types[tf] = 'bullish'
            elif signal < 0:
                current_div_types[tf] = 'bearish'
            else:
                current_div_types[tf] = 'none'
        
        # Calculate consistency score
        match_count = 0
        total_weight = 0
        
        for tf1 in timeframes:
            for tf2 in timeframes:
                if tf1 < tf2 and current_div_types[tf1] == current_div_types[tf2] and current_div_types[tf1] != 'none':
                    match_count += 1
                    total_weight += timeframe_weights[tf1] + timeframe_weights[tf2]
        
        if total_weight > 0:
            consistency_scores.iloc[i] = match_count * total_weight / len(timeframes)
    
    # Persistence analysis
    for tf in timeframes:
        persistence_count = 0
        prev_div_type = 'none'
        
        for i in range(len(data)):
            current_signal = divergence_signals[tf].iloc[i]
            
            if current_signal > 0:
                current_div_type = 'bullish'
            elif current_signal < 0:
                current_div_type = 'bearish'
            else:
                current_div_type = 'none'
            
            if current_div_type == prev_div_type and current_div_type != 'none':
                persistence_count += 1
            else:
                persistence_count = 1 if current_div_type != 'none' else 0
            
            # Cap persistence at 10 days
            persistence_count = min(persistence_count, 10)
            
            # Apply exponential persistence weight (minimum 2-day persistence required)
            if persistence_count >= 2:
                persistence_trackers[tf].iloc[i] = 1.2 ** (persistence_count - 1)
            else:
                persistence_trackers[tf].iloc[i] = 0
            
            prev_div_type = current_div_type
    
    # Cross-timeframe persistence
    cross_persistence = pd.Series(0.0, index=data.index)
    
    for i in range(len(data)):
        if i < max(timeframes):
            continue
            
        avg_persistence = 0
        total_weight = 0
        
        for tf in timeframes:
            avg_persistence += persistence_trackers[tf].iloc[i] * timeframe_weights[tf]
            total_weight += timeframe_weights[tf]
        
        if total_weight > 0:
            avg_persistence /= total_weight
        
        # Combine with consistency score
        cross_persistence.iloc[i] = avg_persistence * consistency_scores.iloc[i]
    
    # Multi-timeframe signal aggregation with persistence
    aggregated_signal = pd.Series(0.0, index=data.index)
    
    for i in range(len(data)):
        if i < max(timeframes):
            continue
            
        weighted_sum = 0
        total_weight = 0
        
        for tf in timeframes:
            signal = divergence_signals[tf].iloc[i]
            persistence = cross_persistence.iloc[i]
            
            # Apply persistence to signal
            enhanced_signal = signal * persistence
            
            weighted_sum += enhanced_signal * timeframe_weights[tf]
            total_weight += timeframe_weights[tf]
        
        if total_weight > 0:
            aggregated_signal.iloc[i] = weighted_sum / total_weight
    
    # Final signal smoothing with EMA(5)
    final_signal = aggregated_signal.ewm(span=5).mean()
    
    # Fill initial NaN values with 0
    final_signal = final_signal.fillna(0)
    
    return final_signal
