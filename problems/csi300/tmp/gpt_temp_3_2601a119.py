import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum-Efficiency Analysis
    # Calculate price momentum for different timeframes
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_21d'] = data['close'] / data['close'].shift(21) - 1
    
    # Compute price change per unit volume for each timeframe
    data['efficiency_3d'] = data['momentum_3d'] / (data['volume'].rolling(window=3).mean() + 1e-8)
    data['efficiency_8d'] = data['momentum_8d'] / (data['volume'].rolling(window=8).mean() + 1e-8)
    data['efficiency_21d'] = data['momentum_21d'] / (data['volume'].rolling(window=21).mean() + 1e-8)
    
    # Detect momentum-efficiency divergence patterns
    data['momentum_divergence'] = (
        (data['momentum_3d'] - data['momentum_21d']) * 
        (data['efficiency_3d'] - data['efficiency_21d'])
    )
    
    # Volume-Amount Structural Analysis
    # Calculate volume momentum
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    
    # Volume persistence via autocorrelation (5-day lag)
    data['volume_autocorr'] = data['volume'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
    )
    
    # Volatility Regime Classification
    # Calculate ATR (Average True Range)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_20d'] = data['tr'].rolling(window=20).mean()
    data['atr_60d'] = data['tr'].rolling(window=60).mean()
    
    # ATR ratio for regime classification
    data['atr_ratio'] = data['atr_20d'] / (data['atr_60d'] + 1e-8)
    
    # Bollinger width for volatility compression
    data['bb_upper'] = data['close'].rolling(window=20).mean() + 2 * data['close'].rolling(window=20).std()
    data['bb_lower'] = data['close'].rolling(window=20).mean() - 2 * data['close'].rolling(window=20).std()
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['close'].rolling(window=20).mean()
    
    # Volatility regime classification
    conditions = [
        data['atr_ratio'] > 1.2,
        data['atr_ratio'] < 0.8
    ]
    choices = [2, 0]  # 2: High, 1: Normal, 0: Low
    data['vol_regime'] = np.select(conditions, choices, default=1)
    
    # Structural Break Detection
    # Volume-price relationship shifts (rolling correlation)
    data['volume_price_corr'] = data['volume'].rolling(window=20).corr(data['close'])
    data['volume_price_corr_change'] = data['volume_price_corr'] - data['volume_price_corr'].shift(5)
    
    # Volume clustering detection (z-score of volume)
    data['volume_zscore'] = (
        data['volume'] - data['volume'].rolling(window=20).mean()
    ) / (data['volume'].rolling(window=20).std() + 1e-8)
    
    # Liquidity-Efficiency Context
    data['spread_proxy'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['price_impact'] = abs(data['close'] - data['open']) / (data['volume'] + 1e-8)
    
    # Regime-Adaptive Integration
    # High Volatility components
    high_vol_component = (
        data['momentum_divergence'] * 0.4 +
        data['volume_zscore'] * 0.3 +
        data['spread_proxy'] * 0.3
    )
    
    # Low Volatility components
    low_vol_component = (
        data['volume_autocorr'] * 0.4 +
        data['efficiency_3d'] * 0.3 +
        data['bb_width'] * 0.3
    )
    
    # Normal Volatility components
    normal_vol_component = (
        data['momentum_8d'] * 0.3 +
        data['volume_momentum_5d'] * 0.25 +
        data['efficiency_8d'] * 0.25 +
        data['price_impact'] * 0.2
    )
    
    # Apply regime weighting
    conditions = [
        data['vol_regime'] == 2,  # High volatility
        data['vol_regime'] == 0   # Low volatility
    ]
    choices = [high_vol_component, low_vol_component]
    data['regime_alpha'] = np.select(conditions, choices, default=normal_vol_component)
    
    # Structural break enhancement
    data['structural_break_enhancement'] = (
        data['volume_price_corr_change'] * 0.6 +
        data['volume_zscore'] * 0.4
    )
    
    # Composite Alpha Generation
    data['final_alpha'] = (
        data['regime_alpha'] * 0.7 +
        data['structural_break_enhancement'] * 0.2 +
        data['price_impact'] * 0.1
    )
    
    # Normalize the final alpha
    alpha_series = data['final_alpha']
    alpha_series = (alpha_series - alpha_series.rolling(window=60).mean()) / (alpha_series.rolling(window=60).std() + 1e-8)
    
    return alpha_series
