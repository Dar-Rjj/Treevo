import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Dimensional Momentum with Microstructure Confirmation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize output series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    min_periods = max(10, len(data))
    if len(data) < min_periods:
        return alpha
    
    # 1. Momentum Convergence Analysis
    # Intraday Efficiency Momentum
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-Timeframe Momentum Alignment
    data['short_term_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_convergence'] = data['short_term_momentum'] * data['medium_term_momentum']
    
    # Momentum Quality Assessment
    data['efficiency_momentum_divergence'] = data['price_efficiency'] - data['short_term_momentum']
    
    # Calculate rolling correlation for timeframe alignment
    momentum_alignment = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            if len(window_data) >= 3:  # Need at least 3 points for correlation
                corr = window_data['short_term_momentum'].corr(window_data['medium_term_momentum'])
                momentum_alignment.append(corr if not np.isnan(corr) else 0)
            else:
                momentum_alignment.append(0)
        else:
            momentum_alignment.append(0)
    data['momentum_alignment_strength'] = momentum_alignment
    
    # 2. Volume-Amount Dynamics
    # Volume Momentum Analysis
    data['volume_median'] = data['volume'].rolling(window=9, min_periods=5).apply(lambda x: np.median(x[:-1]) if len(x) > 1 else np.nan)
    data['volume_momentum'] = data['volume'] / data['volume_median']
    
    # Volume acceleration
    data['volume_acceleration'] = (data['volume'] - 2 * data['volume'].shift(1) + data['volume'].shift(2)) / data['volume'].shift(1).replace(0, np.nan)
    
    # Amount-Based Liquidity Signals
    data['amount_median'] = data['amount'].rolling(window=9, min_periods=5).apply(lambda x: np.median(x[:-1]) if len(x) > 1 else np.nan)
    data['amount_momentum'] = data['amount'] / data['amount_median']
    
    data['flow_efficiency'] = (data['close'] - data['open']) * data['volume'] / data['amount'].replace(0, np.nan)
    data['liquidity_pressure'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume-Amount Convergence
    data['volume_amount_alignment'] = data['volume_momentum'] * data['amount_momentum']
    data['volume_amount_divergence'] = data['volume_momentum'] - data['amount_momentum']
    
    # 3. Range-Volatility Regime Detection
    # Daily Range Analysis
    data['normalized_range'] = (data['high'] - data['low']) / data['close']
    
    range_median = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i]['normalized_range']
            if len(window_data) > 0:
                range_median.append(np.median(window_data))
            else:
                range_median.append(np.nan)
        else:
            range_median.append(np.nan)
    data['range_median_4d'] = range_median
    
    data['range_momentum'] = (data['normalized_range'] - data['range_median_4d']) / data['range_median_4d'].replace(0, np.nan)
    
    # Volatility Breakout Signals
    data['compression_ratio'] = data['normalized_range'] / data['range_median_4d'].replace(0, np.nan)
    data['low_compression'] = (data['compression_ratio'] < 0.7).astype(float)
    
    # Range-Momentum Interaction
    data['range_adjusted_momentum'] = data['momentum_convergence'] * data['normalized_range']
    
    # 4. Order Flow Microstructure
    # Buy-Sell Pressure Analysis
    data['intraday_pressure'] = (data['close'] - data['open']) * data['volume'] / data['amount'].replace(0, np.nan)
    
    # Calculate pressure persistence (rolling correlation)
    pressure_persistence = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            if len(window_data) >= 3:
                price_changes = window_data['close'] - window_data['open']
                corr = price_changes.corr(window_data['volume'])
                pressure_persistence.append(corr if not np.isnan(corr) else 0)
            else:
                pressure_persistence.append(0)
        else:
            pressure_persistence.append(0)
    data['pressure_persistence'] = pressure_persistence
    
    # Market Depth Assessment
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['depth_indicator'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Microstructure-Momentum Integration
    data['pressure_weighted_momentum'] = data['momentum_convergence'] * data['intraday_pressure'].abs()
    
    # 5. Gap Dynamics Integration
    # Overnight Gap Analysis
    data['gap_magnitude'] = (data['open'] / data['close'].shift(1)) - 1
    data['gap_fill_efficiency'] = (data['close'] - data['open']) / data['gap_magnitude'].replace(0, np.nan)
    
    # Gap-Volume Interaction
    data['gap_volume_alignment'] = data['gap_magnitude'] * data['volume_momentum']
    
    # Gap-Momentum Synthesis
    data['gap_enhanced_momentum'] = data['momentum_convergence'] * data['gap_magnitude'].abs()
    
    # 6. Multi-Dimensional Signal Quality and Composite Alpha Generation
    for i in range(len(data)):
        if i < 10:  # Skip initial periods for stability
            alpha.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        
        # Component calculations
        momentum_component = (
            current['momentum_convergence'] * 
            (1 + current['momentum_alignment_strength']) *
            np.sign(current['price_efficiency'])
        )
        
        volume_amount_component = (
            current['volume_amount_alignment'] *
            current['flow_efficiency'] *
            (1 - abs(current['volume_amount_divergence']))
        )
        
        range_volatility_component = (
            current['range_adjusted_momentum'] *
            (1 + current['low_compression']) *
            np.sign(current['range_momentum'])
        )
        
        microstructure_component = (
            current['pressure_weighted_momentum'] *
            current['depth_indicator'] *
            (1 + current['pressure_persistence'])
        )
        
        gap_component = (
            current['gap_enhanced_momentum'] *
            np.sign(current['gap_fill_efficiency']) *
            current['gap_volume_alignment']
        )
        
        # Adaptive weighting based on market regimes
        compression_weight = max(0, 1 - current['compression_ratio'])
        normal_weight = 1 - compression_weight
        volume_convergence_weight = min(1, abs(current['volume_amount_alignment']))
        gap_weight = min(1, abs(current['gap_magnitude']))
        
        # Final composite alpha
        composite_alpha = (
            compression_weight * range_volatility_component +
            normal_weight * (
                0.4 * momentum_component +
                0.3 * volume_amount_component +
                0.2 * microstructure_component +
                0.1 * gap_component
            ) +
            volume_convergence_weight * volume_amount_component +
            gap_weight * gap_component
        )
        
        alpha.iloc[i] = composite_alpha if not np.isnan(composite_alpha) else 0
    
    # Clean extreme values and normalize
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    alpha = (alpha - alpha.rolling(window=20, min_periods=10).mean()) / alpha.rolling(window=20, min_periods=10).std().replace(0, 1)
    
    return alpha
