import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume-Momentum Convergence factor
    Analyzes momentum alignment across short (5-day), medium (10-day), and long (20-day) timeframes
    """
    
    # Calculate basic components
    close = df['close']
    volume = df['volume']
    
    def calculate_momentum_components(price_series, vol_series, period):
        """Calculate price and volume momentum components for a given period"""
        
        # Price momentum
        price_change = price_series.pct_change(period)
        price_acceleration = price_change.diff()
        
        # Volume momentum  
        volume_change = vol_series.pct_change(period)
        volume_acceleration = volume_change.diff()
        
        # Price-volume correlation
        price_vol_corr = price_series.rolling(window=period).corr(vol_series)
        
        # Momentum divergence (price vs volume)
        momentum_divergence = (price_change - volume_change) / (price_change.abs() + volume_change.abs() + 1e-8)
        
        return {
            'price_change': price_change,
            'price_acceleration': price_acceleration,
            'volume_change': volume_change,
            'volume_acceleration': volume_acceleration,
            'price_vol_corr': price_vol_corr,
            'momentum_divergence': momentum_divergence
        }
    
    # Calculate momentum components for all timeframes
    short_term = calculate_momentum_components(close, volume, 5)
    medium_term = calculate_momentum_components(close, volume, 10)
    long_term = calculate_momentum_components(close, volume, 20)
    
    # Cross-Timeframe Convergence Analysis
    def calculate_alignment_score(short, medium, long, component):
        """Calculate alignment strength across timeframes for a given component"""
        s_dir = np.sign(short[component])
        m_dir = np.sign(medium[component])
        l_dir = np.sign(long[component])
        
        # Count agreements
        alignment_count = (s_dir == m_dir).astype(int) + (s_dir == l_dir).astype(int) + (m_dir == l_dir).astype(int)
        
        # Weight by magnitude (average absolute value across timeframes)
        magnitude_weight = (abs(short[component]) + abs(medium[component]) + abs(long[component])) / 3
        
        return alignment_count * magnitude_weight
    
    # Calculate alignment scores for key components
    price_change_alignment = calculate_alignment_score(short_term, medium_term, long_term, 'price_change')
    volume_change_alignment = calculate_alignment_score(short_term, medium_term, long_term, 'volume_change')
    price_accel_alignment = calculate_alignment_score(short_term, medium_term, long_term, 'price_acceleration')
    
    # Divergence pattern recognition
    def detect_divergence_patterns(short, medium, long):
        """Identify conflicting signals and resolution patterns"""
        
        # Price-volume divergence detection
        pv_divergence_short = abs(short['momentum_divergence'])
        pv_divergence_medium = abs(medium['momentum_divergence'])
        pv_divergence_long = abs(long['momentum_divergence'])
        
        # Cross-timeframe divergence (when timeframes disagree on direction)
        price_directions = pd.DataFrame({
            'short': np.sign(short['price_change']),
            'medium': np.sign(medium['price_change']),
            'long': np.sign(long['price_change'])
        })
        
        cross_divergence = price_directions.std(axis=1)
        
        return {
            'pv_divergence': (pv_divergence_short + pv_divergence_medium + pv_divergence_long) / 3,
            'cross_divergence': cross_divergence
        }
    
    divergence_patterns = detect_divergence_patterns(short_term, medium_term, long_term)
    
    # Volume confirmation analysis
    def volume_confirmation_analysis(short, medium, long):
        """Analyze volume trend consistency and breakout confirmation"""
        
        # Volume trend consistency (all timeframes showing same volume direction)
        vol_directions = pd.DataFrame({
            'short': np.sign(short['volume_change']),
            'medium': np.sign(medium['volume_change']),
            'long': np.sign(long['volume_change'])
        })
        
        volume_consistency = (vol_directions.std(axis=1) == 0).astype(float)
        
        # Volume breakout (significant volume increase across multiple timeframes)
        vol_breakout = ((short['volume_change'] > 0.1) & 
                       (medium['volume_change'] > 0.05) & 
                       (long['volume_change'] > 0.02)).astype(float)
        
        return volume_consistency * 0.7 + vol_breakout * 0.3
    
    volume_confirmation = volume_confirmation_analysis(short_term, medium_term, long_term)
    
    # Signal Generation
    def generate_convergence_signal(short, medium, long, alignments, divergences, vol_conf):
        """Generate final convergence signal"""
        
        # Directional bias (majority momentum direction)
        price_directions = pd.DataFrame({
            'short': np.sign(short['price_change']),
            'medium': np.sign(medium['price_change']),
            'long': np.sign(long['price_change'])
        })
        
        directional_bias = price_directions.mode(axis=1)[0].fillna(0)
        
        # Acceleration trend (weighted by timeframe)
        accel_trend = (short['price_acceleration'] * 0.5 + 
                      medium['price_acceleration'] * 0.3 + 
                      long['price_acceleration'] * 0.2)
        
        # Combined convergence score
        convergence_strength = (alignments['price_change'] * 0.4 + 
                               alignments['volume_change'] * 0.3 + 
                               alignments['price_accel'] * 0.3)
        
        # Adjust for divergence (reduce signal strength when divergences exist)
        divergence_penalty = (divergences['pv_divergence'] + divergences['cross_divergence']) / 2
        adjusted_convergence = convergence_strength * (1 - divergence_penalty)
        
        # Final factor with volume confirmation
        final_signal = directional_bias * adjusted_convergence * (0.7 + 0.3 * vol_conf)
        
        return final_signal
    
    alignments = {
        'price_change': price_change_alignment,
        'volume_change': volume_change_alignment,
        'price_accel': price_accel_alignment
    }
    
    # Generate final factor
    factor = generate_convergence_signal(short_term, medium_term, long_term, 
                                       alignments, divergence_patterns, volume_confirmation)
    
    # Normalize and clean
    factor = (factor - factor.rolling(window=60, min_periods=20).mean()) / factor.rolling(window=60, min_periods=20).std()
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return factor
