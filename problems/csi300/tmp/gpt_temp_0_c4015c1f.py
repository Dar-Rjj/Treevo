import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-dimensional Momentum Convergence factor that combines multiple momentum dimensions
    to generate a robust momentum signal
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate various momentum dimensions
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Dimension 1: Absolute price momentum (5, 10, 20 days)
    mom_5d = close.pct_change(5)
    mom_10d = close.pct_change(10)
    mom_20d = close.pct_change(20)
    
    # Dimension 2: Relative momentum (vs rolling mean)
    rolling_mean_10 = close.rolling(window=10).mean()
    rolling_mean_20 = close.rolling(window=20).mean()
    rel_mom_10 = (close - rolling_mean_10) / rolling_mean_10
    rel_mom_20 = (close - rolling_mean_20) / rolling_mean_20
    
    # Dimension 3: Volatility-adjusted momentum
    # Calculate rolling volatility (10-day std)
    vol_10d = close.pct_change().rolling(window=10).std()
    # Avoid division by zero
    vol_10d = vol_10d.replace(0, np.nan)
    
    # Risk-adjusted momentum (similar to Sharpe ratio)
    risk_adj_mom_10 = mom_10d / vol_10d
    risk_adj_mom_20 = mom_20d / vol_10d
    
    # Dimension 4: Volume-confirmed momentum
    volume_ma_5 = volume.rolling(window=5).mean()
    volume_ma_10 = volume.rolling(window=10).mean()
    
    # Volume momentum confirmation
    vol_conf_5 = (volume / volume_ma_5 - 1) * np.sign(mom_5d)
    vol_conf_10 = (volume / volume_ma_10 - 1) * np.sign(mom_10d)
    
    # Dimension 5: Breakout momentum
    # Recent high/low levels
    recent_high_10 = high.rolling(window=10).max()
    recent_low_10 = low.rolling(window=10).min()
    
    # Breakout strength
    breakout_up = (close - recent_high_10.shift(1)) / recent_high_10.shift(1)
    breakout_down = (close - recent_low_10.shift(1)) / recent_low_10.shift(1)
    breakout_momentum = np.where(breakout_up > 0, breakout_up, 
                                np.where(breakout_down < 0, -breakout_down, 0))
    
    # Convergence detection and scoring
    for i in range(len(df)):
        if i < 20:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        # Collect all momentum dimensions for current period
        dimensions = [
            mom_5d.iloc[i], mom_10d.iloc[i], mom_20d.iloc[i],  # Absolute momentum
            rel_mom_10.iloc[i], rel_mom_20.iloc[i],            # Relative momentum
            risk_adj_mom_10.iloc[i], risk_adj_mom_20.iloc[i],  # Risk-adjusted momentum
            vol_conf_5.iloc[i], vol_conf_10.iloc[i],           # Volume-confirmed momentum
            breakout_momentum[i]                               # Breakout momentum
        ]
        
        # Remove NaN values
        valid_dims = [d for d in dimensions if not np.isnan(d)]
        
        if not valid_dims:
            result.iloc[i] = 0
            continue
        
        # Count positive dimensions (convergence strength)
        positive_dims = sum(1 for d in valid_dims if d > 0)
        negative_dims = sum(1 for d in valid_dims if d < 0)
        
        # Calculate convergence ratio
        total_dims = len(valid_dims)
        convergence_ratio = (positive_dims - negative_dims) / total_dims
        
        # Calculate average momentum strength (magnitude of convergence)
        avg_momentum_strength = np.mean([abs(d) for d in valid_dims])
        
        # Calculate dimension correlation (how aligned are the signals)
        positive_strength = np.mean([d for d in valid_dims if d > 0]) if positive_dims > 0 else 0
        negative_strength = np.mean([d for d in valid_dims if d < 0]) if negative_dims > 0 else 0
        
        # Convergence strength score (0-1 scale)
        if positive_dims > negative_dims:
            convergence_strength = positive_dims / total_dims
            directional_strength = positive_strength
        elif negative_dims > positive_dims:
            convergence_strength = negative_dims / total_dims
            directional_strength = negative_strength
        else:
            convergence_strength = 0.5
            directional_strength = 0
        
        # Final factor calculation
        # Weight by convergence strength and directional momentum
        result.iloc[i] = (convergence_ratio * convergence_strength * 
                         directional_strength * avg_momentum_strength)
    
    # Normalize the factor
    result = (result - result.rolling(window=20).mean()) / result.rolling(window=20).std()
    
    return result.fillna(0)
