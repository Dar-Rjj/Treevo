import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum Fracture Divergence factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price and volume features
    df['returns'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Rolling windows for different timeframes
    windows = {'short': 3, 'medium': 10, 'long': 30}
    
    for i in range(max(windows.values()), len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # 1. Calculate Momentum Fracture Components
        # Price Momentum Fracture
        current_data['return_direction'] = np.sign(current_data['returns'])
        current_data['momentum_streak'] = (current_data['return_direction'] == current_data['return_direction'].shift(1)).cumsum()
        current_data['momentum_break'] = (current_data['return_direction'] != current_data['return_direction'].shift(1)).astype(int)
        
        # Volume Momentum Fracture
        volume_median = current_data['volume'].rolling(window=20, min_periods=1).median()
        current_data['volume_above_median'] = (current_data['volume'] > volume_median).astype(int)
        current_data['volume_streak'] = (current_data['volume_above_median'] == current_data['volume_above_median'].shift(1)).cumsum()
        current_data['volume_break'] = (current_data['volume_above_median'] != current_data['volume_above_median'].shift(1)).astype(int)
        
        # 2. Compute Price-Volume Fracture Alignment
        current_data['fracture_alignment'] = (current_data['momentum_break'] & current_data['volume_break']).astype(int)
        current_data['fracture_misalignment'] = (current_data['momentum_break'] != current_data['volume_break']).astype(int)
        
        # 3. Calculate Multi-Timeframe Fracture Signals
        fracture_signals = {}
        for timeframe, window in windows.items():
            # Short-term fracture signals
            if timeframe == 'short':
                fracture_signals[timeframe] = current_data['momentum_break'].rolling(window=window).sum()
            # Medium-term fracture signals
            elif timeframe == 'medium':
                weekly_returns = current_data['close'].pct_change(5)
                fracture_signals[timeframe] = (np.sign(weekly_returns) != np.sign(weekly_returns.shift(1))).rolling(window=window).sum()
            # Long-term fracture signals
            else:
                ma_short = current_data['close'].rolling(window=15).mean()
                ma_long = current_data['close'].rolling(window=30).mean()
                fracture_signals[timeframe] = (np.sign(ma_short - ma_long) != np.sign((ma_short - ma_long).shift(1))).rolling(window=window).sum()
        
        # 4. Detect Fracture-Recovery Patterns
        current_data['recovery_signal'] = 0
        for j in range(1, min(5, len(current_data))):
            if current_data['momentum_break'].iloc[-j] == 1:
                # Check for quick recovery (bounce within 2 days)
                if j <= 2 and current_data['returns'].iloc[-1] > 0:
                    current_data['recovery_signal'].iloc[-1] = 1
                # Check for failed recovery (continued decline)
                elif j <= 3 and current_data['returns'].iloc[-1] < -0.02:
                    current_data['recovery_signal'].iloc[-1] = -1
        
        # 5. Measure Range Expansion Context
        current_data['range_percentile'] = current_data['daily_range'].rolling(window=50).rank(pct=True)
        current_data['range_expansion'] = (current_data['daily_range'] > current_data['daily_range'].rolling(window=20).mean()).astype(int)
        current_data['close_position'] = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low'])
        
        # 6. Detect Fracture Divergence Patterns
        price_fracture_intensity = current_data['momentum_break'].rolling(window=5).sum()
        volume_fracture_intensity = current_data['volume_break'].rolling(window=5).sum()
        
        price_volume_divergence = (price_fracture_intensity - volume_fracture_intensity) / (price_fracture_intensity + volume_fracture_intensity + 1e-8)
        
        # Timeframe divergence
        timeframe_divergence = 0
        if len(fracture_signals) >= 2:
            short_medium_div = fracture_signals['short'].iloc[-1] - fracture_signals['medium'].iloc[-1]
            medium_long_div = fracture_signals['medium'].iloc[-1] - fracture_signals['long'].iloc[-1]
            timeframe_divergence = (short_medium_div + medium_long_div) / 2
        
        # 7. Synthesize Composite Factor
        current_idx = current_data.index[-1]
        
        # Base fracture alignment signal
        alignment_signal = current_data['fracture_alignment'].iloc[-5:].mean()
        
        # Divergence intensity
        divergence_intensity = abs(price_volume_divergence.iloc[-1]) + abs(timeframe_divergence)
        
        # Range expansion scaling
        range_scaling = current_data['range_percentile'].iloc[-1]
        
        # Recovery pattern weighting
        recovery_weight = 1 + 0.5 * current_data['recovery_signal'].iloc[-1]
        
        # Final factor calculation
        factor_value = (
            alignment_signal * 
            divergence_intensity * 
            range_scaling * 
            recovery_weight * 
            np.sign(current_data['returns'].iloc[-1])
        )
        
        factor_values[current_idx] = factor_value
    
    # Fill NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
