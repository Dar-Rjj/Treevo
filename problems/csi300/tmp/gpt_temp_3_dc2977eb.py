import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Microstructure Divergence Factor
    Combines fractal dimension analysis, volume timing patterns, and momentum divergence
    to predict future stock returns.
    """
    df = df.copy()
    
    # Helper function for fractal dimension calculation
    def calculate_fractal_dimension(series, window):
        """Calculate fractal dimension using Higuchi method approximation"""
        fractal_dims = []
        for i in range(len(series)):
            if i < window - 1:
                fractal_dims.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1].values
            if len(window_data) < window:
                fractal_dims.append(np.nan)
                continue
                
            # Simplified Higuchi method approximation
            L = []
            for k in range(1, min(5, window//2)):
                Lk = 0
                for m in range(k):
                    segments = []
                    for j in range(0, len(window_data) - m, k):
                        if j + k < len(window_data):
                            segment = window_data[j + m:j + m + k]
                            if len(segment) > 1:
                                segments.append(np.sum(np.abs(np.diff(segment))))
                    
                    if segments:
                        Lk += np.mean(segments) / k
                
                if k > 0 and Lk > 0:
                    L.append(np.log(Lk))
            
            if len(L) >= 2:
                x = np.log(range(1, len(L) + 1))
                if len(x) == len(L):
                    try:
                        slope = np.polyfit(x, L, 1)[0]
                        fractal_dims.append(-slope if slope != 0 else 1.0)
                    except:
                        fractal_dims.append(1.0)
                else:
                    fractal_dims.append(1.0)
            else:
                fractal_dims.append(1.0)
        
        return pd.Series(fractal_dims, index=series.index)
    
    # 1. Multi-Timeframe Fractal Dimension
    close_prices = df['close']
    
    # Short-term fractal dimension (5-day)
    fd_short = calculate_fractal_dimension(close_prices, 5)
    
    # Medium-term fractal dimension (20-day)
    fd_medium = calculate_fractal_dimension(close_prices, 20)
    
    # Fractal dimension ratio
    fd_ratio = fd_short / fd_medium
    
    # 2. Intraday Momentum Components
    df['prev_close'] = df['close'].shift(1)
    
    # Opening momentum
    opening_momentum = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Midday momentum
    midday_momentum = (df['high'] + df['low']) / 2 - df['open']
    
    # Closing momentum
    closing_momentum = df['close'] - (df['high'] + df['low']) / 2
    
    # 3. Volume Fractal Timing Patterns
    # Simplified volume concentration (assuming first/last hour proxies)
    volume_total = df['volume']
    
    # Volume fractal dimension
    volume_fd = calculate_fractal_dimension(df['volume'], 10)
    
    # Volume clustering intensity (using rolling extremes)
    volume_rolling_max = df['volume'].rolling(window=5, min_periods=1).max()
    volume_rolling_min = df['volume'].rolling(window=5, min_periods=1).min()
    volume_clustering = (df['volume'] - volume_rolling_min) / (volume_rolling_max - volume_rolling_min + 1e-8)
    
    # Volume structure complexity
    volume_complexity = volume_fd * volume_clustering
    
    # Volume timing divergence (simplified using daily patterns)
    morning_volume_proxy = df['volume'].rolling(window=3).apply(lambda x: x.iloc[0] if len(x) == 3 else np.nan)
    afternoon_volume_proxy = df['volume'].rolling(window=3).apply(lambda x: x.iloc[-1] if len(x) == 3 else np.nan)
    
    morning_concentration = morning_volume_proxy / volume_total
    afternoon_intensity = afternoon_volume_proxy / volume_total
    volume_timing_divergence = morning_concentration - afternoon_intensity
    
    # 4. Fractal Momentum Divergence Framework
    # Momentum divergence signals
    opening_vs_closing = opening_momentum - closing_momentum
    
    # Momentum fragmentation score
    momentum_fragmentation = np.abs(opening_vs_closing) / (np.abs(midday_momentum) + 1e-8)
    
    # Volume-weighted momentum
    volume_weighted_momentum = (opening_momentum * morning_concentration + 
                               closing_momentum * afternoon_intensity)
    
    # Fractal-structure weighted momentum
    fractal_weighted_momentum_high = volume_weighted_momentum * fd_ratio
    fractal_weighted_momentum_low = volume_weighted_momentum / (fd_ratio + 1e-8)
    
    # Fractal regime adjustment
    fractal_regime = np.where(fd_ratio > 1.0, fractal_weighted_momentum_high, fractal_weighted_momentum_low)
    
    # 5. Price Efficiency & Fractal Alignment
    # Price efficiency metrics
    range_utilization = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    extreme_price_efficiency = (df['high'] - df['low']) / (np.abs(df['prev_high'] - df['prev_low']) + 1e-8)
    
    intraday_volatility_clustering = range_utilization * extreme_price_efficiency
    
    # Volume-price fractal alignment
    volume_price_correlation = df['volume'].rolling(window=10).corr(close_prices)
    alignment_strength = np.abs(volume_price_correlation)
    
    # 6. Multi-Horizon Convergence Framework
    # Multi-period momentum coherence
    df['close_t_minus_3'] = df['close'].shift(3)
    df['close_t_minus_4'] = df['close'].shift(4)
    
    multi_period_momentum_coherence = (df['close'] - df['close_t_minus_3']) / (
        (df['close'].shift(1) - df['close_t_minus_4']) + 1e-8)
    
    volatility_adjusted_momentum = multi_period_momentum_coherence * intraday_volatility_clustering
    
    # Convergence strength using fractal and microstructure alignment
    convergence_strength = (alignment_strength * fd_ratio * volume_complexity).fillna(0)
    
    # 7. Final Alpha Signal Construction
    # Base signals
    fragmentation_momentum = opening_vs_closing * volume_timing_divergence
    volume_timing_momentum = volume_weighted_momentum * momentum_fragmentation
    efficiency_momentum = volatility_adjusted_momentum * range_utilization
    
    # Composite divergence signal
    composite_divergence = (fragmentation_momentum + 
                           volume_timing_momentum + 
                           efficiency_momentum)
    
    # Apply convergence strength multiplier
    alpha_signal = composite_divergence * convergence_strength
    
    # Clean up and return
    alpha_signal = alpha_signal.replace([np.inf, -np.inf], np.nan)
    alpha_signal = alpha_signal.fillna(method='ffill').fillna(0)
    
    return alpha_signal
