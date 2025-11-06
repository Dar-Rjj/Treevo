import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Price-Volume Fractality with Microstructure Asymmetry
    Combines multi-scale fractal analysis with microstructure patterns to generate alpha signals
    """
    
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Scale Fractal Dimension Calculation
    def compute_fractal_dimension(series, window=20):
        """Calculate fractal dimension using Hurst exponent approximation"""
        # Calculate price range and volatility
        price_range = series.rolling(window).max() - series.rolling(window).min()
        volatility = series.rolling(window).std()
        
        # Avoid division by zero
        volatility = volatility.replace(0, np.nan)
        
        # Fractal dimension approximation (1.5 - Hurst exponent)
        hurst_approx = np.log(price_range / volatility) / np.log(window)
        fractal_dim = 1.5 - hurst_approx.clip(-0.5, 1.5)
        
        return fractal_dim
    
    # Compute multi-scale fractal dimensions
    data['fractal_1min'] = compute_fractal_dimension(data['close'], window=5)
    data['fractal_5min'] = compute_fractal_dimension(data['close'], window=15)
    data['fractal_daily'] = compute_fractal_dimension(data['close'], window=20)
    
    # Volume fractal properties
    data['volume_fractal'] = compute_fractal_dimension(data['volume'], window=15)
    
    # 2. Microstructure Asymmetry Patterns
    def calculate_microstructure_asymmetry(data, window=10):
        """Calculate microstructure asymmetry from price and volume patterns"""
        
        # Price-based microstructure signals
        price_change = data['close'].pct_change()
        high_low_range = (data['high'] - data['low']) / data['close']
        
        # Volume-weighted price moves
        volume_weighted_returns = (price_change * data['volume']).rolling(window).mean()
        
        # Trade size asymmetry (using amount as proxy for trade size)
        large_trade_bias = (data['amount'].rolling(window).quantile(0.8) / 
                           data['amount'].rolling(window).quantile(0.2))
        
        # Intraday price pressure
        open_close_strength = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
        
        # Combine microstructure signals
        micro_asymmetry = (volume_weighted_returns * large_trade_bias * 
                          open_close_strength.rolling(window).mean())
        
        return micro_asymmetry
    
    data['micro_asymmetry'] = calculate_microstructure_asymmetry(data)
    
    # 3. Fractal-Regime Transition Detection
    def detect_fractal_regime_changes(data):
        """Detect transitions between fractal regimes"""
        
        # Fractal dimension changes
        fractal_change_1min = data['fractal_1min'].diff(3)
        fractal_change_5min = data['fractal_5min'].diff(5)
        
        # Multi-scale fractal convergence/divergence
        fractal_divergence = (data['fractal_1min'] - data['fractal_5min']).abs()
        
        # Regime change signals
        smooth_to_rough = ((fractal_change_1min > 0.1) & 
                          (fractal_change_5min > 0.05)).astype(float)
        
        rough_to_smooth = ((fractal_change_1min < -0.1) & 
                          (fractal_change_5min < -0.05)).astype(float)
        
        regime_transition = smooth_to_rough - rough_to_smooth
        
        return regime_transition, fractal_divergence
    
    regime_transition, fractal_divergence = detect_fractal_regime_changes(data)
    data['regime_transition'] = regime_transition
    data['fractal_divergence'] = fractal_divergence
    
    # 4. Price Gaps and Continuity Fractals
    def calculate_gap_fractals(data):
        """Calculate gap-related fractal properties"""
        
        # Overnight gaps
        overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        # Gap persistence
        gap_fill_speed = overnight_gap.rolling(5).std()
        
        # Intraday continuity
        intraday_continuity = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
        
        # Gap clustering indicator
        gap_clustering = overnight_gap.rolling(10).apply(lambda x: (x > 0.01).sum() - (x < -0.01).sum())
        
        return overnight_gap.abs(), gap_fill_speed, intraday_continuity, gap_clustering
    
    gap_size, gap_fill_speed, intraday_continuity, gap_clustering = calculate_gap_fractals(data)
    data['gap_size'] = gap_size
    data['gap_fill_speed'] = gap_fill_speed
    data['intraday_continuity'] = intraday_continuity
    data['gap_clustering'] = gap_clustering
    
    # 5. Generate Fractal-Microstructure Alpha Signals
    def generate_alpha_signals(data):
        """Combine fractal and microstructure signals into alpha factor"""
        
        # Signal components
        fractal_strength = (data['fractal_1min'] + data['fractal_5min'] + data['fractal_daily']) / 3
        micro_strength = data['micro_asymmetry'].rolling(10).mean()
        
        # Regime-adaptive signal weighting
        high_fractal_regime = (fractal_strength > fractal_strength.rolling(50).quantile(0.7))
        low_fractal_regime = (fractal_strength < fractal_strength.rolling(50).quantile(0.3))
        
        # High fractal dimension with microstructure confirmation
        high_fractal_signal = (high_fractal_regime * 
                              micro_strength * 
                              (1 - data['fractal_divergence']))
        
        # Low fractal dimension with regime transition signals
        low_fractal_signal = (low_fractal_regime * 
                             data['regime_transition'] * 
                             data['gap_fill_speed'])
        
        # Gap-based signals in different fractal regimes
        gap_signals_high_fractal = (high_fractal_regime * 
                                   data['gap_clustering'] * 
                                   data['intraday_continuity'])
        
        gap_signals_low_fractal = (low_fractal_regime * 
                                  (-data['gap_size']) * 
                                  data['intraday_continuity'])
        
        # Combine all signals with regime-adaptive weights
        alpha_signal = (
            high_fractal_signal.fillna(0) * 0.4 +
            low_fractal_signal.fillna(0) * 0.3 +
            gap_signals_high_fractal.fillna(0) * 0.15 +
            gap_signals_low_fractal.fillna(0) * 0.15
        )
        
        # Normalize the final signal
        alpha_normalized = (alpha_signal - alpha_signal.rolling(100).mean()) / alpha_signal.rolling(100).std()
        
        return alpha_normalized
    
    # Generate final alpha factor
    alpha_factor = generate_alpha_signals(data)
    
    return alpha_factor
