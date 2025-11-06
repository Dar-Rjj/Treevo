import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Asymmetric Volatility-Regime Momentum
    # Calculate Raw Momentum
    momentum_10d = df['close'].pct_change(periods=10)
    
    # Detect Volatility Regime
    high_low_range = df['high'] - df['low']
    avg_range_20d = high_low_range.rolling(window=20).mean()
    volatility_regime = high_low_range > avg_range_20d
    
    # Calculate Downside Volatility (for high volatility mode)
    negative_returns = np.where(df['close'] < df['close'].shift(1), 
                               (df['low'] - df['close'].shift(1)) / df['close'].shift(1), 0)
    downside_vol = pd.Series(negative_returns, index=df.index).rolling(window=10).std()
    
    # Calculate Upside Volatility (for low volatility mode)
    positive_returns = np.where(df['close'] > df['close'].shift(1), 
                               (df['high'] - df['close'].shift(1)) / df['close'].shift(1), 0)
    upside_vol = pd.Series(positive_returns, index=df.index).rolling(window=10).std()
    
    # Apply Regime-Specific Adjustments
    adjusted_momentum = np.where(volatility_regime,
                                momentum_10d / (downside_vol + 1e-8),
                                momentum_10d / (upside_vol + 1e-8))
    
    # Incorporate Volume Confirmation
    volume_percentile = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)) if len(x) == 20 else np.nan
    )
    volume_filtered_momentum = adjusted_momentum * volume_percentile
    
    # Price-Volume Fractal Divergence
    # Calculate Price Fractal Dimension (simplified Hurst)
    price_range = df['high'] - df['low']
    price_hurst = np.log(price_range.rolling(window=10).std() + 1e-8) / np.log(10)
    
    # Calculate Volume Fractal Dimension
    volume_changes = df['volume'].pct_change().abs()
    volume_hurst = np.log(volume_changes.rolling(window=10).std() + 1e-8) / np.log(10)
    
    # Measure Fractal Divergence
    fractal_divergence = price_hurst - volume_hurst
    divergence_correlation = fractal_divergence.rolling(window=10).corr(momentum_10d)
    
    # Generate Complexity Signal
    # Calculate consecutive same-sign returns
    returns = df['close'].pct_change()
    sign_changes = returns.rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])])
    )
    
    # Calculate intraday efficiency
    intraday_return = (df['close'] - df['open']).abs()
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    efficiency_ratio = intraday_return / (true_range + 1e-8)
    
    complexity_signal = divergence_correlation * sign_changes * efficiency_ratio
    
    # Liquidity-Enhanced Gap Filling
    # Calculate Opening Gap
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Incorporate Liquidity Signal
    volume_to_amount = df['volume'] / (df['amount'] + 1e-8)
    
    # Measure Historical Filling Tendency
    gap_closures = opening_gap.rolling(window=20).apply(
        lambda x: len([gap for gap in x if abs(gap) < 0.01]) / len(x) if len(x) == 20 else np.nan
    )
    
    # Generate Filling Signal
    volume_percentile_fill = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 60)) if len(x) == 20 else np.nan
    )
    filling_signal = opening_gap * gap_closures * volume_percentile_fill
    
    # Volatility Compression Momentum
    # Measure Range Compression
    range_compression = high_low_range / (high_low_range.rolling(window=10).mean() + 1e-8)
    
    # Track Volume During Compression
    volume_anomaly = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-8)
    
    # Calculate Directional Bias
    intraday_returns = (df['close'] - df['open']) / df['open']
    directional_bias = intraday_returns.rolling(window=5).mean()
    
    # Generate Expansion Signal
    recent_momentum = df['close'].pct_change(periods=5)
    expansion_signal = (1 / range_compression) * volume_anomaly * directional_bias * recent_momentum
    
    # Accumulation Distribution Efficiency
    # Calculate Accumulation Distribution Line
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
    money_flow_volume = money_flow_multiplier * df['volume']
    adl = money_flow_volume.cumsum()
    
    # Compute ADL Momentum
    adl_momentum = adl.pct_change(periods=5)
    adl_direction_persistence = (adl.diff() > 0).rolling(window=5).sum()
    
    # Measure Price Movement Efficiency
    true_range_eff = np.maximum(df['high'] - df['low'], 
                               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
    efficiency_ratio_adl = (df['close'] - df['open']).abs() / (true_range_eff + 1e-8)
    
    # Combine Signals
    volume_percentile_adl = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 65)) if len(x) == 20 else np.nan
    )
    adl_direction_weight = np.where(adl.diff() > 0, 1.2, 0.8)
    adl_signal = adl_momentum * efficiency_ratio_adl * volume_percentile_adl * adl_direction_weight
    
    # Combine all factors with equal weighting
    final_factor = (volume_filtered_momentum.fillna(0) + 
                   complexity_signal.fillna(0) + 
                   filling_signal.fillna(0) + 
                   expansion_signal.fillna(0) + 
                   adl_signal.fillna(0))
    
    return pd.Series(final_factor, index=df.index)
