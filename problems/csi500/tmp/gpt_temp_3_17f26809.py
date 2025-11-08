import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Regime-Aware Volume-Price Divergence
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    # Price Momentum Acceleration
    mom_5 = data['close'] / data['close'].shift(5) - 1
    mom_10 = data['close'] / data['close'].shift(10) - 1
    mom_20 = data['close'] / data['close'].shift(20) - 1
    
    # Momentum acceleration (change rate)
    accel_5 = mom_5 - mom_5.shift(5)
    accel_10 = mom_10 - mom_10.shift(10)
    accel_20 = mom_20 - mom_20.shift(20)
    
    # Volume-Price Divergence
    vol_mom_5 = data['volume'] / data['volume'].shift(5) - 1
    vol_mom_10 = data['volume'] / data['volume'].shift(10) - 1
    vol_mom_20 = data['volume'] / data['volume'].shift(20) - 1
    
    # Divergence signals
    div_5 = mom_5 - vol_mom_5
    div_10 = mom_10 - vol_mom_10
    div_20 = mom_20 - vol_mom_20
    
    # Regime Detection Using Amount Data
    # Amount-based regime
    amount_20ma = data['amount'].rolling(window=20).mean()
    amount_accel = data['amount'] / data['amount'].shift(5) - 1
    
    # Amount regime classification
    amount_regime = pd.cut(amount_accel, 
                          bins=[-np.inf, -0.1, 0.1, np.inf], 
                          labels=['low', 'neutral', 'high'])
    
    # Volatility assessment
    price_range = (data['high'] - data['low']) / data['close']
    vol_20 = price_range.rolling(window=20).mean()
    
    # Market condition classification
    vol_regime = pd.cut(vol_20, 
                       bins=[-np.inf, vol_20.quantile(0.33), vol_20.quantile(0.66), np.inf], 
                       labels=['low_vol', 'medium_vol', 'high_vol'])
    
    # Exponential Smoothing Application
    alpha = 0.3
    
    # Smoothed momentum acceleration
    smooth_accel_5 = accel_5.ewm(alpha=alpha).mean()
    smooth_accel_10 = accel_10.ewm(alpha=alpha).mean()
    smooth_accel_20 = accel_20.ewm(alpha=alpha).mean()
    
    # Smoothed divergence
    smooth_div_5 = div_5.ewm(alpha=alpha).mean()
    smooth_div_10 = div_10.ewm(alpha=alpha).mean()
    smooth_div_20 = div_20.ewm(alpha=alpha).mean()
    
    # Combined momentum and divergence signals
    momentum_signal = (smooth_accel_5 + smooth_accel_10 + smooth_accel_20) / 3
    divergence_signal = (smooth_div_5 + smooth_div_10 + smooth_div_20) / 3
    
    # Cross-Sectional Ranking and Volatility Normalization
    # Note: For single stock implementation, we'll use rolling percentiles
    momentum_rank = momentum_signal.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    divergence_rank = divergence_signal.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Combined ranking score
    combined_rank = (momentum_rank + divergence_rank) / 2
    
    # Volatility normalization
    returns_20d = data['close'].pct_change(20)
    vol_20d = returns_20d.rolling(window=20).std()
    
    # Risk-adjusted factor values
    risk_adjusted_rank = combined_rank / (vol_20d + 1e-8)
    
    # Adaptive Factor Construction
    # Regime-weighted signal combination
    regime_weights = pd.Series(index=data.index, dtype=float)
    
    # High participation regimes: emphasize volume confirmation
    high_participation = (amount_regime == 'high') & (vol_regime.isin(['low_vol', 'medium_vol']))
    regime_weights[high_participation] = 0.7 * momentum_rank + 0.3 * divergence_rank
    
    # Low participation regimes: emphasize price momentum
    low_participation = (amount_regime == 'low') | (vol_regime == 'high_vol')
    regime_weights[low_participation] = 0.3 * momentum_rank + 0.7 * divergence_rank
    
    # Neutral regimes: balanced approach
    neutral_conditions = ~high_participation & ~low_participation
    regime_weights[neutral_conditions] = 0.5 * momentum_rank + 0.5 * divergence_rank
    
    # Multi-timeframe integration with acceleration strength weighting
    accel_strength = (abs(smooth_accel_5) + abs(smooth_accel_10) + abs(smooth_accel_20)) / 3
    accel_weight = accel_strength / (accel_strength.rolling(window=20).mean() + 1e-8)
    
    # Final alpha factor construction
    alpha_factor = regime_weights * np.tanh(accel_weight)
    
    # Final volatility normalization
    final_alpha = alpha_factor / (vol_20d + 1e-8)
    
    return final_alpha
