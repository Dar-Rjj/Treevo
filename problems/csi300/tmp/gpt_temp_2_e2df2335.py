import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for volatility estimation
    data['returns'] = data['close'].pct_change()
    
    # Multi-Timeframe Momentum Component
    # Calculate momentum at different timeframes
    mom_short = (data['close'] / data['close'].shift(5) - 1)
    mom_medium = (data['close'] / data['close'].shift(13) - 1)
    mom_long = (data['close'] / data['close'].shift(34) - 1)
    
    # Volatility adjustment using 20-day rolling standard deviation of returns
    volatility = data['returns'].rolling(window=20, min_periods=10).std()
    volatility = volatility.replace(0, np.nan)  # Avoid division by zero
    
    # Volatility-adjusted momentum
    mom_short_adj = mom_short / volatility
    mom_medium_adj = mom_medium / volatility
    mom_long_adj = mom_long / volatility
    
    # Efficiency-Convergence Component
    # Open-High Efficiency
    high_open_diff = data['high'] - data['open']
    open_high_eff = high_open_diff / high_open_diff.shift(1)
    open_high_eff_trend = open_high_eff.rolling(window=5).mean()
    
    # Open-Low Efficiency
    open_low_diff = data['open'] - data['low']
    open_low_eff = open_low_diff / open_low_diff.shift(1)
    open_low_eff_trend = open_low_eff.rolling(window=5).mean()
    
    # Close-Close Efficiency (momentum efficiency)
    close_eff = data['close'].pct_change() / data['close'].pct_change().shift(1)
    close_eff_trend = close_eff.rolling(window=5).mean()
    
    # Efficiency convergence strength
    eff_convergence = (open_high_eff_trend + open_low_eff_trend + close_eff_trend) / 3
    
    # Detect efficiency-momentum divergence
    momentum_regime = (mom_short_adj + mom_medium_adj + mom_long_adj) / 3
    efficiency_divergence = eff_convergence - momentum_regime.rolling(window=5).mean()
    
    # Volume-Volatility Confirmation
    # Volume momentum calculations
    vol_short_trend = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    vol_medium_trend = data['volume'].rolling(window=15).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 15 else np.nan
    )
    
    # Volume-price divergence
    price_momentum = data['close'].pct_change(5)
    volume_divergence = vol_short_trend - price_momentum.rolling(window=5).mean()
    
    # Volatility clustering analysis
    volatility_persistence = volatility.rolling(window=10).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 10 else np.nan
    )
    
    # Volume-volatility confirmation signal
    vol_vol_confirmation = (vol_short_trend * volatility_persistence).fillna(0)
    
    # Cross-Timeframe Signal Integration
    # Exponential weighting for multi-timeframe momentum
    weights = np.array([0.5, 0.3, 0.2])  # Short, medium, long weights
    weighted_momentum = (mom_short_adj * weights[0] + 
                        mom_medium_adj * weights[1] + 
                        mom_long_adj * weights[2])
    
    # Apply efficiency convergence as multiplier
    eff_multiplier = 1 + np.tanh(efficiency_divergence * 0.1)
    
    # Apply volume-volatility confirmation filter
    vol_filter = np.tanh(vol_vol_confirmation * 0.05) + 1
    
    # Final alpha factor calculation
    alpha_factor = (weighted_momentum * eff_multiplier * vol_filter)
    
    # Clean and return the factor
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
