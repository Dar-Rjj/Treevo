import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate composite alpha factor using price momentum, volume interaction, 
    range dynamics, volatility patterns, and multi-factor integration.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic components
    close = data['close']
    open_price = data['open']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Price Momentum & Reversal components
    normalized_price_change = (close - close.shift(1)) / (high - low).replace(0, np.nan)
    intraday_momentum_strength = (close - open_price) / (high - low).replace(0, np.nan)
    
    # Multi-day price consistency (5-day window)
    price_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            window_data = []
            for j in range(i-4, i+1):
                if j > 0 and (high.iloc[j] - low.iloc[j]) > 0:
                    price_change = close.iloc[j] - close.iloc[j-1] if j > 0 else 0
                    window_data.append(price_change / (high.iloc[j] - low.iloc[j]))
            if window_data:
                price_consistency.iloc[i] = sum(window_data)
    
    # Volume-Price Interaction components
    volume_weighted_price_move = (close - close.shift(1)) * volume / (high - low).replace(0, np.nan)
    volume_change_impact = np.sign(close - close.shift(1)) * (volume - volume.shift(1)) / volume.shift(1).replace(0, np.nan)
    
    # Volume-price correlation (5-day window)
    volume_price_corr = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            corr_sum = 0
            for j in range(i-4, i+1):
                if j > 0:
                    price_sign = np.sign(close.iloc[j] - close.iloc[j-1])
                    volume_sign = np.sign(volume.iloc[j] - volume.iloc[j-1])
                    corr_sum += price_sign * volume_sign
            volume_price_corr.iloc[i] = corr_sum
    
    # Price Range Dynamics components
    range_efficiency = abs(close - open_price) / (high - low).replace(0, np.nan)
    range_breakout_signal = (close - high.shift(1)) / (high.shift(1) - low.shift(1)).replace(0, np.nan)
    range_stability = (high - low) / (high.shift(1) - low.shift(1)).replace(0, np.nan)
    
    # Volatility Patterns components
    price_volatility_clustering = abs(close - close.shift(1)) / abs(close.shift(1) - close.shift(2)).replace(0, np.nan)
    range_based_volatility = (high - low) / close.shift(1).replace(0, np.nan)
    
    # Volatility persistence (5-day window)
    volatility_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            vol_sum = 0
            for j in range(i-4, i+1):
                if j > 0 and (high.iloc[j] - low.iloc[j]) > 0:
                    vol_sum += abs(close.iloc[j] - close.iloc[j-1]) / (high.iloc[j] - low.iloc[j])
            volatility_persistence.iloc[i] = vol_sum
    
    # Composite Alpha Signals
    momentum_volume_composite = normalized_price_change * (volume / volume.shift(1)).replace([np.inf, -np.inf], np.nan)
    range_volume_composite = range_based_volatility * np.sqrt(volume)
    
    # Multi-factor integration (5-day window)
    multi_factor_integration = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            factor_sum = 0
            for j in range(i-4, i+1):
                if j > 0 and (high.iloc[j] - low.iloc[j]) > 0:
                    factor_sum += (close.iloc[j] - close.iloc[j-1]) * volume.iloc[j] / (high.iloc[j] - low.iloc[j])
            multi_factor_integration.iloc[i] = factor_sum
    
    # Combine all factors with equal weighting
    factor_components = [
        normalized_price_change,
        intraday_momentum_strength,
        price_consistency,
        volume_weighted_price_move,
        volume_change_impact,
        volume_price_corr,
        range_efficiency,
        range_breakout_signal,
        range_stability,
        price_volatility_clustering,
        range_based_volatility,
        volatility_persistence,
        momentum_volume_composite,
        range_volume_composite,
        multi_factor_integration
    ]
    
    # Standardize and combine factors
    for i in range(len(data)):
        if i >= 4:  # Ensure we have enough history
            valid_components = []
            for comp in factor_components:
                if not pd.isna(comp.iloc[i]):
                    valid_components.append(comp.iloc[i])
            
            if valid_components:
                # Z-score normalization using historical data up to current point
                historical_data = []
                for comp in factor_components:
                    if len(comp) > i:
                        historical_data.extend(comp.iloc[max(0, i-20):i+1].dropna().tolist())
                
                if historical_data:
                    mean_val = np.mean(historical_data)
                    std_val = np.std(historical_data) if np.std(historical_data) > 0 else 1
                    factor.iloc[i] = np.mean([(x - mean_val) / std_val for x in valid_components])
    
    return factor.fillna(0)
