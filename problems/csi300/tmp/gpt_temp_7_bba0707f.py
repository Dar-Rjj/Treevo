import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Regime Detection with Fractal Volume Dynamics
    """
    data = df.copy()
    
    # Price Regime Identification
    # Volatility Regime Components
    returns = data['close'].pct_change()
    abs_returns = returns.abs()
    
    # 5-day volatility clustering
    vol_clustering_5d = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            vol_clustering = 0
            for j in range(i-4, i):
                if j+1 < len(data):
                    vol_clustering += abs_returns.iloc[j] * abs_returns.iloc[j+1]
            vol_clustering_5d.iloc[i] = vol_clustering
    
    # 10-day volatility persistence
    vol_persistence_10d = pd.Series(index=data.index, dtype=float)
    for i in range(9, len(data)):
        if i >= 9:
            vol_persistence = 0
            for j in range(i-9, i-1):
                if j+2 < len(data):
                    vol_persistence += abs_returns.iloc[j] * abs_returns.iloc[j+2]
            vol_persistence_10d.iloc[i] = vol_persistence
    
    # Regime stability ratio
    regime_stability = vol_persistence_10d / (vol_clustering_5d + 1e-8)
    
    # Momentum Fractal Structure
    # Short-term momentum cascade
    short_momentum = ((data['close'] / data['close'].shift(1) - 1) * 
                     (data['close'].shift(1) / data['close'].shift(2) - 1) * 
                     (data['close'].shift(2) / data['close'].shift(3) - 1))
    
    # Medium-term momentum cascade
    medium_momentum = ((data['close'] / data['close'].shift(3) - 1) * 
                      (data['close'].shift(3) / data['close'].shift(6) - 1) * 
                      (data['close'].shift(6) / data['close'].shift(9) - 1))
    
    # Fractal momentum ratio
    fractal_momentum = short_momentum / (medium_momentum + 1e-8)
    
    # Volume Fractal Dynamics
    # Multi-Scale Volume Patterns
    volume_returns = data['volume'].pct_change() + 1
    
    # Volume fractal short-term
    vol_fractal_short = (volume_returns * 
                        volume_returns.shift(1) * 
                        volume_returns.shift(2))
    
    # Volume fractal medium-term
    vol_fractal_medium = ((data['volume'] / data['volume'].shift(3)) * 
                         (data['volume'].shift(3) / data['volume'].shift(6)) * 
                         (data['volume'].shift(6) / data['volume'].shift(9)))
    
    # Volume fractal ratio
    vol_fractal_ratio = vol_fractal_short / (vol_fractal_medium + 1e-8)
    
    # Volume-Price Fractal Alignment
    # Short-term alignment
    short_alignment = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        if i >= 2:
            vol_changes = []
            price_changes = []
            for j in range(i-2, i+1):
                if j > 0:
                    vol_changes.append(data['volume'].iloc[j] / data['volume'].iloc[j-1])
                    price_changes.append(data['close'].iloc[j] / data['close'].iloc[j-1])
            if len(vol_changes) > 1:
                short_alignment.iloc[i] = np.corrcoef(vol_changes, price_changes)[0,1]
    
    # Medium-term alignment
    medium_alignment = pd.Series(index=data.index, dtype=float)
    for i in range(8, len(data)):
        if i >= 8:
            vol_changes = []
            price_changes = []
            for j in range(i-8, i+1):
                if j > 0:
                    vol_changes.append(data['volume'].iloc[j] / data['volume'].iloc[j-1])
                    price_changes.append(data['close'].iloc[j] / data['close'].iloc[j-1])
            if len(vol_changes) > 1:
                medium_alignment.iloc[i] = np.corrcoef(vol_changes, price_changes)[0,1]
    
    # Alignment gradient
    alignment_gradient = medium_alignment - short_alignment
    
    # Intraday Fractal Structure
    # Opening-Closing Fractal Patterns
    opening_fractal = ((data['open'] / data['close'].shift(1) - 1) * 
                      (data['open'].shift(1) / data['close'].shift(2) - 1) * 
                      (data['open'].shift(2) / data['close'].shift(3) - 1))
    
    closing_fractal = ((data['close'] / data['open'] - 1) * 
                      (data['close'].shift(1) / data['open'].shift(1) - 1) * 
                      (data['close'].shift(2) / data['open'].shift(2) - 1))
    
    intraday_fractal_ratio = opening_fractal / (closing_fractal + 1e-8)
    
    # High-Low Fractal Dynamics
    high_fractal = ((data['high'] / data['close'].shift(1) - 1) * 
                   (data['high'].shift(1) / data['close'].shift(2) - 1) * 
                   (data['high'].shift(2) / data['close'].shift(3) - 1))
    
    low_fractal = ((data['close'].shift(1) / data['low'] - 1) * 
                  (data['close'].shift(2) / data['low'].shift(1) - 1) * 
                  (data['close'].shift(3) / data['low'].shift(2) - 1))
    
    range_fractal_ratio = high_fractal / (low_fractal + 1e-8)
    
    # Cross-Fractal Correlation
    # Price-Volume Fractal Correlation
    short_fractal_corr = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        if i >= 2:
            price_changes = []
            volume_changes = []
            for j in range(i-2, i+1):
                if j > 0:
                    price_changes.append(data['close'].iloc[j] / data['close'].iloc[j-1])
                    volume_changes.append(data['volume'].iloc[j] / data['volume'].iloc[j-1])
            if len(price_changes) > 1:
                short_fractal_corr.iloc[i] = np.corrcoef(price_changes, volume_changes)[0,1]
    
    medium_fractal_corr = pd.Series(index=data.index, dtype=float)
    for i in range(8, len(data)):
        if i >= 8:
            price_changes = []
            volume_changes = []
            for j in range(i-8, i+1):
                if j > 0:
                    price_changes.append(data['close'].iloc[j] / data['close'].iloc[j-1])
                    volume_changes.append(data['volume'].iloc[j] / data['volume'].iloc[j-1])
            if len(price_changes) > 1:
                medium_fractal_corr.iloc[i] = np.corrcoef(price_changes, volume_changes)[0,1]
    
    fractal_corr_gradient = medium_fractal_corr - short_fractal_corr
    
    # Intra-Interday Fractal Alignment
    intraday_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            open_ratios = []
            close_ratios = []
            for j in range(i-4, i+1):
                if j > 0:
                    open_ratios.append(data['open'].iloc[j] / data['close'].iloc[j-1])
                    close_ratios.append(data['close'].iloc[j] / data['open'].iloc[j])
            if len(open_ratios) > 1:
                intraday_consistency.iloc[i] = np.corrcoef(open_ratios, close_ratios)[0,1]
    
    interday_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            current_returns = []
            next_returns = []
            for j in range(i-4, i):
                if j+1 < len(data):
                    current_returns.append(data['close'].iloc[j] / data['close'].iloc[j-1])
                    next_returns.append(data['close'].iloc[j+1] / data['close'].iloc[j])
            if len(current_returns) > 1:
                interday_consistency.iloc[i] = np.corrcoef(current_returns, next_returns)[0,1]
    
    temporal_alignment = intraday_consistency / (interday_consistency + 1e-8)
    
    # Regime-Aware Signal Construction
    # Core Fractal Integration
    base_fractal = fractal_momentum * vol_fractal_ratio
    intraday_enhanced = base_fractal * intraday_fractal_ratio
    range_adjusted = intraday_enhanced * range_fractal_ratio
    
    # Regime-Based Refinement
    volatility_filtered = range_adjusted * regime_stability
    correlation_enhanced = volatility_filtered * fractal_corr_gradient
    temporal_filtered = correlation_enhanced * temporal_alignment
    
    # Alpha Generation
    raw_alpha = temporal_filtered * alignment_gradient
    refined_alpha = raw_alpha * vol_fractal_ratio
    
    # Clean up infinite values and NaNs
    refined_alpha = refined_alpha.replace([np.inf, -np.inf], np.nan)
    refined_alpha = refined_alpha.fillna(0)
    
    return refined_alpha
