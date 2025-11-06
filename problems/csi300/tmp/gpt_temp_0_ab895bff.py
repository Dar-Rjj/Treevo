import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Structure Analysis
    # Intraday Volatility Patterns
    daily_range = df['high'] - df['low']
    open_close_range = abs(df['open'] - df['close'])
    
    # Opening vs Closing Volatility Ratio (5-day rolling)
    vol_ratio = (open_close_range.rolling(window=5).std() / 
                daily_range.rolling(window=5).std()).fillna(1)
    
    # Volatility clustering using 5-day variance of daily returns
    returns = df['close'].pct_change()
    vol_clustering = returns.rolling(window=5).var().fillna(0)
    
    # Multi-Timeframe Volatility Regimes
    # Volatility persistence (autocorrelation of daily ranges)
    range_autocorr = daily_range.rolling(window=10).apply(
        lambda x: x.autocorr(lag=1), raw=False
    ).fillna(0)
    
    # Regime transition detection
    vol_ma_short = daily_range.rolling(window=5).mean()
    vol_ma_long = daily_range.rolling(window=20).mean()
    vol_regime = np.where(vol_ma_short > vol_ma_long, 1, 0)  # 1=high, 0=low
    
    # Price-Volume Synergy Metrics
    # Directional Volume Effectiveness
    up_days = returns > 0
    up_volume = df['volume'].where(up_days, 0)
    total_up_volume = up_volume.rolling(window=10).sum()
    total_volume = df['volume'].rolling(window=10).sum()
    vol_effectiveness = (total_up_volume / total_volume).fillna(0.5)
    
    # Volume-Price Divergence (10-day correlation)
    price_change = returns.rolling(window=5).std()
    volume_change = df['volume'].pct_change().rolling(window=5).std()
    vol_price_divergence = price_change.rolling(window=10).corr(volume_change).fillna(0)
    
    # Trade Impact Analysis
    # Price impact per unit volume (normalized)
    price_impact = (abs(returns) / (df['volume'] + 1e-10)).rolling(window=10).mean().fillna(0)
    
    # Volume concentration effectiveness
    volume_rank = df['volume'].rolling(window=20).rank(pct=True)
    high_volume_periods = volume_rank > 0.9
    high_volume_returns = abs(returns).where(high_volume_periods, 0)
    vol_concentration = (high_volume_returns.rolling(window=20).sum() / 
                        abs(returns).rolling(window=20).sum()).fillna(0)
    
    # Adaptive Signal Framework
    # Volatility-adjusted trend strength
    momentum = df['close'].pct_change(periods=5)
    realized_vol = returns.rolling(window=10).std()
    vol_adjusted_trend = (momentum / (realized_vol + 1e-10)).fillna(0)
    
    # Compression-expansion ratio
    compression_ratio = (daily_range / daily_range.rolling(window=20).mean()).fillna(1)
    
    # Dynamic factor selection based on volatility regime
    high_vol_factors = (
        vol_adjusted_trend * 0.4 +
        vol_effectiveness * 0.3 +
        price_impact * 0.3
    )
    
    low_vol_factors = (
        compression_ratio * 0.4 +
        vol_price_divergence * 0.3 +
        vol_concentration * 0.3
    )
    
    # Regime-based factor selection
    adaptive_factor = np.where(vol_regime == 1, high_vol_factors, low_vol_factors)
    
    # Synergy Integration Engine
    # Volatility-Volume-Price alignment
    vol_alignment = (vol_ratio * range_autocorr * vol_clustering).fillna(0)
    
    # Adaptive weighting mechanism
    vol_weight = np.where(vol_regime == 1, 0.6, 0.4)
    price_weight = np.where(vol_regime == 1, 0.3, 0.5)
    synergy_weight = 0.1
    
    # Final factor integration
    final_factor = (
        adaptive_factor * vol_weight +
        vol_alignment * price_weight +
        (vol_effectiveness * vol_price_divergence) * synergy_weight
    )
    
    # Signal quality assessment
    signal_strength = (final_factor / (realized_vol + 1e-10)).fillna(0)
    
    return pd.Series(final_factor * signal_strength, index=df.index)
