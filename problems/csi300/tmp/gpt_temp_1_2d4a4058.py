import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily price ranges
    data['daily_range'] = data['high'] - data['low']
    
    # 1. Regime-Dependent Momentum Structure
    # Compute volatility-adjusted momentum components
    data['short_term_mom'] = (data['close'] - data['close'].shift(3)) / data['daily_range'].rolling(window=4).std()
    data['medium_term_mom'] = (data['close'] - data['close'].shift(10)) / data['daily_range'].rolling(window=11).std()
    data['long_term_mom'] = (data['close'] - data['close'].shift(21)) / data['daily_range'].rolling(window=22).std()
    
    # Calculate momentum hierarchy divergence
    data['mom_divergence'] = (data['short_term_mom'] - data['long_term_mom']).abs() * np.sign(data['medium_term_mom'] - data['short_term_mom'])
    
    # 2. Volume Distribution Asymmetry
    # Since we don't have intraday data, use daily volume patterns
    # Calculate volume concentration (approximation using rolling statistics)
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_concentration'] = data['volume_5d_avg'] / data['volume_20d_avg']
    
    # Volume-price timing relationship
    data['price_move_direction'] = np.sign(data['close'] - data['open'])
    data['volume_price_timing'] = data['volume'] * data['price_move_direction'] / data['volume'].rolling(window=10).std()
    
    # 3. Multi-Timeframe Volatility Regime Shifts
    # Calculate volatility regime persistence
    data['volatility_5d'] = data['daily_range'].rolling(window=5).std()
    data['volatility_20d'] = data['daily_range'].rolling(window=20).std()
    
    # Volatility autocorrelation (5-day)
    data['vol_autocorr'] = data['daily_range'].rolling(window=6).apply(
        lambda x: x[:-1].corr(x[1:]) if len(x) == 6 and not x.isna().any() else np.nan, 
        raw=False
    )
    
    # Volatility mean-reversion strength
    data['vol_mean_reversion'] = (data['volatility_20d'] - data['volatility_5d']) / data['volatility_20d']
    
    # Cross-timeframe volatility alignment
    data['vol_momentum_divergence'] = (data['volatility_5d'] - data['volatility_5d'].shift(5)) - (data['volatility_20d'] - data['volatility_20d'].shift(5))
    
    # Volatility structure consistency
    data['vol_structure_slope'] = (data['volatility_5d'] - data['volatility_20d']) / data['volatility_20d']
    
    # 4. Generate Adaptive Momentum Composite
    # Volatility regime strength
    data['vol_regime_strength'] = data['vol_autocorr'].abs() * (1 - data['vol_mean_reversion'].abs())
    
    # Combine signals with regime-dependent weighting
    # Base momentum signal
    momentum_signal = data['mom_divergence'].fillna(0)
    
    # Volume timing adjustment
    volume_adjustment = data['volume_price_timing'].fillna(0) * data['volume_concentration'].fillna(1)
    
    # Volatility structure alignment
    vol_alignment = -data['vol_momentum_divergence'].fillna(0) * data['vol_structure_slope'].fillna(0)
    
    # Dynamic signal conditioning based on volatility regime
    regime_weight = 1 + data['vol_regime_strength'].fillna(0)
    
    # Generate composite factor
    composite_factor = (
        momentum_signal * 0.4 +
        volume_adjustment * 0.3 +
        vol_alignment * 0.3
    ) * regime_weight
    
    # Apply smoothing and normalization
    factor = composite_factor.rolling(window=5, min_periods=1).mean()
    factor = (factor - factor.rolling(window=20, min_periods=1).mean()) / factor.rolling(window=20, min_periods=1).std()
    
    return factor
