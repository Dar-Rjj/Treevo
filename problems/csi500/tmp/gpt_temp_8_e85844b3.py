import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Liquidity Divergence
    # Short-term divergence: (Close(t)/Close(t-3)-1) - (Volume(t)/Volume(t-3)-1)
    short_divergence = (data['close'] / data['close'].shift(3) - 1) - (data['volume'] / data['volume'].shift(3) - 1)
    
    # Medium-term divergence: (Close(t)/Close(t-5)-1) - (Volume(t)/Volume(t-5)-1)
    medium_divergence = (data['close'] / data['close'].shift(5) - 1) - (data['volume'] / data['volume'].shift(5) - 1)
    
    # Divergence momentum: Short-term divergence(t) - Short-term divergence(t-1)
    divergence_momentum = short_divergence - short_divergence.shift(1)
    
    # Fractal divergence alignment: Correlation between short and medium-term divergence
    fractal_div_alignment = short_divergence.rolling(window=10, min_periods=5).corr(medium_divergence)
    
    # Liquidity Momentum Geometry
    # Amount momentum: Amount(t)/Amount(t-5)-1
    amount_momentum = data['amount'] / data['amount'].shift(5) - 1
    
    # Volume efficiency: Volume(t)/(High(t)-Low(t))
    volume_efficiency = data['volume'] / (data['high'] - data['low'])
    
    # Liquidity acceleration: Amount momentum × Volume efficiency
    liquidity_acceleration = amount_momentum * volume_efficiency
    
    # Multi-scale liquidity persistence: Amount momentum(t) - Amount momentum(t-3)
    liquidity_persistence = amount_momentum - amount_momentum.shift(3)
    
    # Divergence-Gap Coupling
    # Gap absorption divergence: (Close(t)-Open(t))/|Open(t)-Close(t-1)| × Divergence strength
    gap_absorption_divergence = ((data['close'] - data['open']) / 
                                np.abs(data['open'] - data['close'].shift(1))) * short_divergence
    
    # Multi-scale gap divergence: Short-term divergence × Gap efficiency
    gap_efficiency = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    multi_scale_gap_divergence = short_divergence * gap_efficiency
    
    # Divergence momentum with gap persistence: Divergence momentum × Gap persistence
    gap_persistence = gap_efficiency - gap_efficiency.shift(3)
    divergence_momentum_gap = divergence_momentum * gap_persistence
    
    # Fractal Efficiency Dynamics with Liquidity
    # Intraday efficiency: |Close(t) - Open(t)| / (High(t) - Low(t))
    intraday_efficiency = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Short-term efficiency: |Close(t) - Close(t-3)| / Σ|Close(i) - Close(i-1)| for i=t-2 to t
    short_efficiency_numerator = np.abs(data['close'] - data['close'].shift(3))
    short_efficiency_denominator = (np.abs(data['close'] - data['close'].shift(1)) + 
                                   np.abs(data['close'].shift(1) - data['close'].shift(2)) + 
                                   np.abs(data['close'].shift(2) - data['close'].shift(3)))
    short_term_efficiency = short_efficiency_numerator / short_efficiency_denominator
    
    # Medium-term efficiency: |Close(t) - Close(t-5)| / Σ|Close(i) - Close(i-1)| for i=t-4 to t
    medium_efficiency_numerator = np.abs(data['close'] - data['close'].shift(5))
    medium_efficiency_denominator = sum(np.abs(data['close'].shift(i) - data['close'].shift(i+1)) 
                                       for i in range(5))
    medium_term_efficiency = medium_efficiency_numerator / medium_efficiency_denominator
    
    # Efficiency-liquidity coupling: Efficiency ratio × Volume efficiency
    efficiency_liquidity_coupling = intraday_efficiency * volume_efficiency
    
    # Multi-Scale Volume Entropy with Microstructure
    # Volume fractal ratios
    volume_fractal_1 = data['volume'] / data['volume'].shift(1)
    volume_fractal_2 = data['volume'] / data['volume'].shift(3)
    volume_fractal_3 = data['volume'] / data['volume'].shift(5)
    
    # Volume entropy calculation
    volume_fractals = pd.DataFrame({
        'f1': volume_fractal_1,
        'f2': volume_fractal_2,
        'f3': volume_fractal_3
    })
    
    def calculate_entropy(row):
        valid_values = [x for x in row if pd.notna(x) and x > 0]
        if len(valid_values) < 2:
            return np.nan
        total = sum(valid_values)
        probabilities = [v/total for v in valid_values]
        entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
        return entropy
    
    volume_entropy = volume_fractals.apply(calculate_entropy, axis=1)
    
    # Microstructure Volume Patterns
    # Price impact: |Close(t)-Open(t)|/Volume(t)
    price_impact = np.abs(data['close'] - data['open']) / data['volume']
    
    # Volume clustering: Volume(t)/Volume(t-1) - Volume(t-1)/Volume(t-2)
    volume_clustering = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Microstructure momentum: Price impact × Volume clustering
    microstructure_momentum = price_impact * volume_clustering
    
    # Volume dispersion: MAD(Volume[t-5:t]) / Median(Volume[t-5:t])
    def volume_dispersion_calc(series):
        if len(series) < 5:
            return np.nan
        mad = np.median(np.abs(series - np.median(series)))
        median_val = np.median(series)
        return mad / median_val if median_val != 0 else np.nan
    
    volume_dispersion = data['volume'].rolling(window=6, min_periods=5).apply(volume_dispersion_calc, raw=False)
    
    # Momentum Geometry with Liquidity Confirmation
    # Intraday Momentum Trajectory
    # Morning momentum: (High(t) - Open(t)) / (Open(t) - Low(t))
    morning_momentum = (data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)
    
    # Afternoon momentum: (Close(t) - Low(t)) / (High(t) - Close(t))
    afternoon_momentum = (data['close'] - data['low']) / (data['high'] - data['close']).replace(0, np.nan)
    
    # Momentum asymmetry: Morning momentum - Afternoon momentum
    momentum_asymmetry = morning_momentum - afternoon_momentum
    
    # Trajectory curvature: (Close(t) - (High(t)+Low(t))/2) / (High(t) - Low(t))
    trajectory_curvature = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-Scale Momentum with Liquidity
    # Short-term momentum: Close(t)/Close(t-3) - 1
    short_momentum = data['close'] / data['close'].shift(3) - 1
    
    # Medium-term momentum: Close(t)/Close(t-5) - 1
    medium_momentum = data['close'] / data['close'].shift(5) - 1
    
    # Liquidity-confirmed momentum: Momentum × Volume efficiency
    liquidity_confirmed_momentum = short_momentum * volume_efficiency
    
    # Composite Alpha Construction
    # Core components weighted by their significance
    alpha_factor = (
        # Multi-scale divergence weighted by gap efficiency
        0.3 * short_divergence * gap_efficiency +
        
        # Liquidity-confirmed divergence with volume entropy
        0.25 * short_divergence * volume_efficiency * (1 + volume_entropy) +
        
        # Divergence persistence adjusted for fractal regime
        0.15 * divergence_momentum * fractal_div_alignment +
        
        # Efficiency-enhanced divergence continuation signals
        0.1 * short_divergence * intraday_efficiency +
        
        # Volume entropy-divergence alignment multiplier
        0.08 * volume_entropy * short_divergence +
        
        # Multi-scale microstructure patterns
        0.06 * microstructure_momentum * short_divergence +
        
        # Intraday trajectory analysis with microstructure
        0.03 * trajectory_curvature * microstructure_momentum +
        
        # Multi-scale momentum confirmation with liquidity
        0.02 * short_momentum * volume_efficiency +
        
        # Geometric efficiency factors with volume efficiency
        0.01 * intraday_efficiency * volume_efficiency
    )
    
    # Apply persistence and consistency filters
    # 3-day divergence consistency
    divergence_consistency = short_divergence.rolling(window=3, min_periods=2).std()
    alpha_factor = alpha_factor * (1 / (1 + divergence_consistency))
    
    # Efficiency trend persistence filter
    efficiency_trend = intraday_efficiency.rolling(window=5, min_periods=3).mean()
    alpha_factor = alpha_factor * (1 + 0.1 * (intraday_efficiency - efficiency_trend))
    
    # Volume dispersion filtering
    alpha_factor = alpha_factor * (1 / (1 + volume_dispersion))
    
    return alpha_factor
