import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Liquidity Momentum Convergence factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Liquidity Pressure Analysis
    # Short-term liquidity pressure (5-day)
    data['directional_pressure'] = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['volume_weighted_pressure'] = data['directional_pressure'] * data['volume']
    data['pressure_momentum_5d'] = data['volume_weighted_pressure'].rolling(window=5, min_periods=3).sum()
    
    # Medium-term liquidity absorption (20-day)
    data['intraday_liquidity'] = (data['high'] - data['low']) / (data['volume'] + 1e-8)
    data['absorption_imbalance'] = data['intraday_liquidity'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    )
    data['absorption_trend_20d'] = data['absorption_imbalance'].rolling(window=5, min_periods=3).mean()
    
    # Long-term liquidity regime (60-day)
    data['volume_clustering'] = data['volume'].rolling(window=60, min_periods=30).apply(
        lambda x: (x - x.mean()).abs().sum() / (x.std() + 1e-8)
    )
    data['regime_persistence'] = data['volume_clustering'].rolling(window=20, min_periods=10).std()
    
    # Volume-Validated Momentum Efficiency
    # Multi-timeframe momentum quality
    data['momentum_5d'] = (data['close'] / data['close'].shift(5) - 1) * data['volume']
    data['momentum_20d'] = (data['close'] / data['close'].shift(20) - 1) * data['volume']
    data['momentum_60d'] = (data['close'] / data['close'].shift(60) - 1) * data['volume']
    
    # Range efficiency assessment
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_persistence'] = data['daily_efficiency'].rolling(window=10, min_periods=5).std()
    
    # Momentum-liquidity alignment
    data['pressure_momentum_corr'] = data['pressure_momentum_5d'].rolling(window=20, min_periods=10).corr(
        data['momentum_5d'].rolling(window=20, min_periods=10).mean()
    )
    data['alignment_score'] = (1 - data['pressure_momentum_corr'].abs()) * np.sign(data['pressure_momentum_corr'])
    
    # Regime Classification & Adaptation
    # Volatility-liquidity regime detection
    data['price_volatility'] = data['close'].pct_change().abs().rolling(window=20, min_periods=10).std()
    data['volume_volatility'] = data['volume'].pct_change().abs().rolling(window=20, min_periods=10).std()
    data['volatility_regime'] = (data['price_volatility'] / (data['volume_volatility'] + 1e-8)).rolling(window=10, min_periods=5).mean()
    
    # Range compression with liquidity analysis
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['range_compression'] = data['price_range'].rolling(window=10, min_periods=5).std()
    data['compression_liquidity'] = data['range_compression'] / (data['intraday_liquidity'] + 1e-8)
    
    # Market regime classification
    data['regime_classification'] = (
        data['volatility_regime'].rolling(window=5, min_periods=3).mean() +
        data['compression_liquidity'].rolling(window=5, min_periods=3).mean() +
        data['alignment_score'].rolling(window=5, min_periods=3).mean()
    ) / 3
    
    # Convergence Pattern Synthesis
    # Multi-timeframe liquidity convergence
    liquidity_convergence = (
        data['pressure_momentum_5d'].rolling(window=5, min_periods=3).mean() +
        data['absorption_trend_20d'].rolling(window=5, min_periods=3).mean() +
        data['regime_persistence'].rolling(window=5, min_periods=3).mean()
    ) / 3
    
    # Momentum-efficiency convergence
    momentum_convergence = (
        data['momentum_5d'].rolling(window=5, min_periods=3).mean() +
        data['momentum_20d'].rolling(window=5, min_periods=3).mean() +
        (1 - data['efficiency_persistence']).rolling(window=5, min_periods=3).mean()
    ) / 3
    
    # Cross-dimension regime convergence
    cross_convergence = (
        liquidity_convergence.rolling(window=5, min_periods=3).mean() +
        momentum_convergence.rolling(window=5, min_periods=3).mean() +
        data['regime_classification'].rolling(window=5, min_periods=3).mean()
    ) / 3
    
    # Adaptive Factor Construction
    # Base signal generation
    base_signal = (
        liquidity_convergence * momentum_convergence * 
        data['alignment_score'].rolling(window=5, min_periods=3).mean()
    )
    
    # Regime-adaptive scaling
    regime_weight = 1 / (1 + data['volatility_regime'].abs().rolling(window=10, min_periods=5).mean())
    compression_weight = 1 / (1 + data['compression_liquidity'].abs().rolling(window=10, min_periods=5).mean())
    
    # Final factor construction
    factor = (
        base_signal * 
        regime_weight * 
        compression_weight * 
        data['daily_efficiency'].rolling(window=5, min_periods=3).mean()
    )
    
    # Risk-aware implementation with liquidity pressure as conviction proxy
    conviction_adjustment = data['pressure_momentum_5d'].rolling(window=10, min_periods=5).std()
    final_factor = factor / (conviction_adjustment + 1e-8)
    
    return final_factor
