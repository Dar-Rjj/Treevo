import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Cross-Asset Liquidity Spillover Alpha factor that captures multiple dimensions
    of liquidity dynamics including spillover effects, spread momentum, and regime transitions.
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Implicit Bid-Ask Spread Estimation from OHLC
    data['spread_est'] = (data['high'] - data['low']) / data['close']
    
    # 2. Spread Momentum Component
    data['spread_momentum'] = data['spread_est'].rolling(window=5).mean() / data['spread_est'].rolling(window=20).mean()
    
    # 3. Volume-Volatility Cointegration Deviation
    # Calculate volatility (using Parkinson estimator for efficiency)
    data['volatility'] = np.log(data['high'] / data['low']) ** 2 / (4 * np.log(2))
    data['volume_norm'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Rolling cointegration relationship (20-day window)
    def rolling_coint_deviation(vol, vol_norm, window=20):
        deviations = []
        for i in range(len(vol)):
            if i < window:
                deviations.append(0)
                continue
            start_idx = i - window + 1
            end_idx = i + 1
            
            # Simple linear relationship between log volume and volatility
            vol_window = vol.iloc[start_idx:end_idx]
            vol_norm_window = vol_norm.iloc[start_idx:end_idx]
            
            if len(vol_window) < 5:  # Minimum observations
                deviations.append(0)
                continue
                
            # Calculate equilibrium relationship
            try:
                slope = np.cov(vol_window, vol_norm_window)[0,1] / np.var(vol_norm_window)
                equilibrium = slope * vol_norm_window.iloc[-1]
                deviation = vol_window.iloc[-1] - equilibrium
                deviations.append(deviation)
            except:
                deviations.append(0)
        
        return pd.Series(deviations, index=vol.index)
    
    data['vol_vol_coint_dev'] = rolling_coint_deviation(data['volatility'], data['volume_norm'])
    
    # 4. Microstructure Noise Ratio
    data['overnight_gap'] = np.abs(data['open'] / data['close'].shift(1) - 1)
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['noise_ratio'] = data['overnight_gap'] / (data['intraday_range'] + 1e-8)
    
    # 5. Liquidity Provision Asymmetry
    # Estimate large trade impact using amount/volume ratio
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['liquidity_absorption'] = data['avg_trade_size'] / data['avg_trade_size'].rolling(window=20).mean()
    
    # 6. Cross-Horizon Liquidity Momentum
    short_term_liq = data['volume'].rolling(window=5).mean()
    medium_term_liq = data['volume'].rolling(window=20).mean()
    data['liquidity_momentum_conv'] = short_term_liq / medium_term_liq
    
    # 7. Liquidity Regime Identification
    # Use volatility and volume to identify regimes
    vol_rank = data['volatility'].rolling(window=20).apply(lambda x: stats.rankdata(x)[-1]/len(x) if len(x) == 20 else 0.5)
    vol_liq_rank = data['volume'].rolling(window=20).apply(lambda x: stats.rankdata(x)[-1]/len(x) if len(x) == 20 else 0.5)
    
    # Regime score: high volatility + low volume = stress regime
    data['regime_score'] = vol_rank - vol_liq_rank
    
    # 8. Combine components with regime-adaptive weighting
    # Normalize components
    components = ['spread_momentum', 'vol_vol_coint_dev', 'noise_ratio', 
                 'liquidity_absorption', 'liquidity_momentum_conv']
    
    for comp in components:
        if comp in data.columns:
            data[f'{comp}_norm'] = (data[comp] - data[comp].rolling(window=20).mean()) / (data[comp].rolling(window=20).std() + 1e-8)
    
    # Regime-dependent weights
    # Stress regimes: emphasize spread momentum and noise ratio
    # Normal regimes: emphasize cointegration and liquidity momentum
    stress_weight = np.where(data['regime_score'] > 0.3, 1.0, 0.0)
    normal_weight = 1 - stress_weight
    
    # Calculate final alpha factor
    if all(f'{comp}_norm' in data.columns for comp in components):
        data['alpha_factor'] = (
            stress_weight * (0.4 * data['spread_momentum_norm'] + 0.3 * data['noise_ratio_norm'] + 0.3 * data['liquidity_absorption_norm']) +
            normal_weight * (0.4 * data['vol_vol_coint_dev_norm'] + 0.4 * data['liquidity_momentum_conv_norm'] + 0.2 * data['spread_momentum_norm'])
        )
    else:
        # Fallback simple combination
        available_comps = [f'{comp}_norm' for comp in components if f'{comp}_norm' in data.columns]
        if available_comps:
            data['alpha_factor'] = data[available_comps].mean(axis=1)
        else:
            data['alpha_factor'] = 0
    
    # Clean up and return
    alpha_series = data['alpha_factor'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return alpha_series
