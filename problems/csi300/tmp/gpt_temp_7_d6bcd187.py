import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Fractal Microstructure Momentum Alpha combining multi-scale fractality analysis
    with microstructure momentum persistence and liquidity absorption measures.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Required minimum data points for calculations
    min_periods = 22
    
    for i in range(min_periods, len(data)):
        current_data = data.iloc[:i+1].copy()
        
        # 1. Multi-Scale Fractal Analysis
        fractal_signals = []
        fractal_consistency = []
        
        # Calculate Hurst exponent for different windows
        windows = [3, 8, 21]
        hurst_values = []
        
        for window in windows:
            if len(current_data) >= window:
                # Calculate Hurst using rescaled range analysis
                hurst = calculate_hurst_exponent(
                    current_data['close'].iloc[-window:],
                    current_data['high'].iloc[-window:],
                    current_data['low'].iloc[-window:]
                )
                hurst_values.append(hurst)
                
                # Fractal breakdown detection
                breakdown_flag = detect_fractal_breakdown(
                    current_data['close'].iloc[-window:],
                    hurst
                )
                fractal_signals.append(breakdown_flag)
        
        # Fractal consistency across timeframes
        if len(hurst_values) >= 2:
            fractal_consistency_score = 1 - np.std(hurst_values) / np.mean(hurst_values)
        else:
            fractal_consistency_score = 0.5
        
        # 2. Microstructure Momentum Persistence
        momentum_persistence = calculate_momentum_persistence(
            current_data['high'].iloc[-5:],
            current_data['low'].iloc[-5:],
            current_data['close'].iloc[-5:],
            current_data['volume'].iloc[-5:]
        )
        
        # 3. Liquidity Momentum Absorption
        liquidity_absorption = calculate_liquidity_absorption(
            current_data['close'].iloc[-8:],
            current_data['volume'].iloc[-8:],
            current_data['amount'].iloc[-8:]
        )
        
        # 4. Combine signals with multi-scale weighting
        if len(fractal_signals) > 0:
            fractal_signal = np.mean(fractal_signals)
        else:
            fractal_signal = 0
        
        # Core alpha calculation
        base_alpha = fractal_signal * momentum_persistence
        adjusted_alpha = base_alpha * (1 - liquidity_absorption)
        
        # Apply multi-scale weighting
        final_alpha = adjusted_alpha * fractal_consistency_score
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial values with neutral signal
    alpha = alpha.fillna(0)
    
    return alpha

def calculate_hurst_exponent(close, high, low):
    """Calculate Hurst exponent using rescaled range analysis."""
    if len(close) < 5:
        return 0.5
    
    try:
        # Calculate log returns
        returns = np.log(close / close.shift(1)).dropna()
        
        if len(returns) < 4:
            return 0.5
            
        # Calculate mean-adjusted returns
        mean_return = returns.mean()
        adjusted_returns = returns - mean_return
        
        # Calculate cumulative deviation
        cumulative_deviation = adjusted_returns.cumsum()
        
        # Calculate range
        R = cumulative_deviation.max() - cumulative_deviation.min()
        
        # Calculate standard deviation
        S = returns.std()
        
        # Avoid division by zero
        if S == 0:
            return 0.5
            
        # Rescaled range
        RS = R / S
        
        # Simple Hurst estimation
        if RS > 0:
            hurst = np.log(RS) / np.log(len(returns))
            return max(0.1, min(0.9, hurst))
        else:
            return 0.5
            
    except:
        return 0.5

def detect_fractal_breakdown(close_series, hurst):
    """Detect when price series loses fractal properties."""
    if len(close_series) < 5:
        return 0
    
    # Calculate volatility
    volatility = close_series.pct_change().std()
    
    # Calculate trend strength
    if len(close_series) >= 3:
        x = np.arange(len(close_series))
        slope, _, r_value, _, _ = linregress(x, close_series.values)
        trend_strength = abs(r_value)
    else:
        trend_strength = 0
    
    # Fractal breakdown occurs when:
    # - Hurst exponent deviates significantly from 0.5 (random walk)
    # - Combined with abnormal volatility or trend patterns
    hurst_deviation = abs(hurst - 0.5)
    
    # Breakdown signal increases with deviation from random walk
    # and abnormal market conditions
    breakdown_signal = hurst_deviation * (1 + volatility) * (1 + trend_strength)
    
    return min(1.0, breakdown_signal)

def calculate_momentum_persistence(high, low, close, volume):
    """Calculate microstructure momentum persistence."""
    if len(close) < 3:
        return 0.5
    
    try:
        # Intraday momentum measures
        intraday_range = (high - low) / close
        close_to_close_returns = close.pct_change().dropna()
        
        if len(close_to_close_returns) < 2:
            return 0.5
        
        # Momentum persistence ratio
        positive_moves = (close_to_close_returns > 0).astype(int)
        momentum_persistence_ratio = positive_moves.rolling(3).mean().iloc[-1]
        
        if pd.isna(momentum_persistence_ratio):
            momentum_persistence_ratio = 0.5
        
        # Volume-weighted momentum efficiency
        volume_weighted_returns = (close_to_close_returns * volume.iloc[1:]).sum()
        total_volume = volume.iloc[1:].sum()
        
        if total_volume > 0:
            volume_efficiency = abs(volume_weighted_returns / total_volume)
        else:
            volume_efficiency = 0
        
        # Combine measures
        persistence_score = 0.6 * momentum_persistence_ratio + 0.4 * volume_efficiency
        
        return max(0, min(1, persistence_score))
        
    except:
        return 0.5

def calculate_liquidity_absorption(close, volume, amount):
    """Calculate how efficiently liquidity absorbs momentum."""
    if len(close) < 3:
        return 0.5
    
    try:
        # Price moves relative to volume
        returns = close.pct_change().dropna()
        volume_changes = volume.pct_change().dropna()
        
        if len(returns) < 2 or len(volume_changes) < 2:
            return 0.5
        
        # Calculate momentum survival rate
        large_moves = abs(returns) > returns.std()
        survived_moves = large_moves.rolling(2).sum().dropna() == 2
        
        if len(survived_moves) > 0:
            survival_rate = survived_moves.mean()
        else:
            survival_rate = 0.5
        
        # Volume efficiency in absorbing moves
        if volume_changes.std() > 0:
            volume_absorption = 1 - (abs(returns.corr(volume_changes)) if not pd.isna(returns.corr(volume_changes)) else 0.5)
        else:
            volume_absorption = 0.5
        
        # Combined liquidity absorption measure
        absorption_score = 0.7 * (1 - survival_rate) + 0.3 * volume_absorption
        
        return max(0, min(1, absorption_score))
        
    except:
        return 0.5
