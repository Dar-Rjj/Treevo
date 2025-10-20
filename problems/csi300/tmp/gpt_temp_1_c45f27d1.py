import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Efficiency-Momentum Core
    # Price efficiency
    price_eff = (data['close'] - data['open']) / (data['high'] - data['low'])
    price_eff = price_eff.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume efficiency
    vol_eff = data['volume'] / (data['high'] - data['low'])
    vol_eff = vol_eff.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Combined efficiency
    combined_eff = price_eff * vol_eff
    
    # Momentum spectrum
    close_ratio_t1 = data['close'] / data['close'].shift(1)
    close_ratio_t8 = data['close'] / data['close'].shift(8)
    momentum_spectrum = (close_ratio_t1 - close_ratio_t8) / np.abs(close_ratio_t8)
    momentum_spectrum = momentum_spectrum.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Efficiency-weighted momentum
    eff_weighted_momentum = combined_eff * momentum_spectrum
    
    # Regime Detection
    # Trend regime condition
    price_change_10d = np.abs(data['close'] / data['close'].shift(10) - 1)
    
    # Calculate sign consistency for trend detection
    close_returns = data['close'].pct_change()
    sign_consistency = close_returns.rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 5 else 0
    )
    
    trend_regime = (price_change_10d > 0.05) & (sign_consistency >= 3)
    
    # Range regime condition
    high_low_range = data['high'] - data['low']
    range_volatility = high_low_range.rolling(window=5, min_periods=1).std() / high_low_range.rolling(window=5, min_periods=1).mean()
    
    # Count days with small price changes relative to range
    small_change_count = ((np.abs(data['close'].diff()) / high_low_range) < 0.3).rolling(window=5, min_periods=1).sum()
    
    range_regime = (range_volatility < 0.5) & (small_change_count >= 3)
    
    # Regime-Specific Enhancement
    # Trend factor
    trend_factor = eff_weighted_momentum * price_change_10d * sign_consistency
    
    # Range factor
    range_factor = eff_weighted_momentum * price_eff * small_change_count
    
    # Signal Refinement
    # Volume confirmation
    vol_ratio = data['volume'] / data['volume'].shift(5)
    close_ratio_1d = data['close'] / data['close'].shift(1)
    volume_confirmation = 1 + np.sign(vol_ratio) * np.sign(close_ratio_1d)
    
    # Factor persistence (using trend factor as base for consistency check)
    factor_sign_consistency = trend_factor.rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 3 else 0
    )
    factor_persistence = 1 + factor_sign_consistency / 3
    
    # Alpha Synthesis
    # Regime selection
    regime_factor = np.where(trend_regime, trend_factor, 
                           np.where(range_regime, range_factor, eff_weighted_momentum))
    
    # Volume adjustment
    volume_adjusted = regime_factor * volume_confirmation
    
    # Persistence enhancement
    final_factor = volume_adjusted * factor_persistence
    
    return pd.Series(final_factor, index=data.index)
