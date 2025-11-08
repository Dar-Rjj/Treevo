import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Regime Momentum Factor combining market regime detection with multi-timeframe momentum analysis
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Market Regime Detection
    # Volatility Regime
    data['volatility_proxy'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['hist_vol_median'] = data['volatility_proxy'].rolling(window=20, min_periods=10).median()
    data['vol_regime_strength'] = data['volatility_proxy'] / data['hist_vol_median']
    data['high_vol_regime'] = (data['vol_regime_strength'] > 1.2).astype(int)
    data['low_vol_regime'] = (data['vol_regime_strength'] < 0.8).astype(int)
    
    # Volume Regime
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_intensity'] = data['volume'] / data['volume_10d_avg']
    data['volume_momentum'] = data['volume'].pct_change(periods=5)
    data['high_volume_regime'] = (data['volume_intensity'] > 1.2).astype(int)
    data['low_volume_regime'] = (data['volume_intensity'] < 0.8).astype(int)
    
    # Liquidity Regime
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['depth_proxy'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_variance'] = data['depth_proxy'].rolling(window=10, min_periods=5).var()
    data['high_liquidity_regime'] = (data['depth_proxy'] > data['depth_proxy'].rolling(20).median()).astype(int)
    
    # Multi-Timeframe Momentum Calculation
    # Short-term momentum (1-3 days)
    data['price_momentum_1d'] = data['close'].pct_change(periods=1)
    data['volume_weighted_momentum'] = data['price_momentum_1d'] * data['volume']
    data['momentum_acceleration'] = data['price_momentum_1d'] - data['price_momentum_1d'].shift(3)
    
    # Medium-term momentum (5-10 days)
    data['price_momentum_5d'] = data['close'].pct_change(periods=5)
    data['amount_confirmed_momentum'] = data['price_momentum_5d'] * data['amount']
    
    # Trend strength (consecutive same-direction days)
    data['price_direction'] = np.sign(data['price_momentum_1d'])
    data['trend_strength'] = data['price_direction'].rolling(window=5).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) == 5 else np.nan, raw=False
    )
    
    # Momentum consistency
    data['timeframe_alignment'] = np.sign(data['price_momentum_1d']) * np.sign(data['price_momentum_5d'])
    data['magnitude_ratio'] = data['price_momentum_1d'].abs() / data['price_momentum_5d'].abs().replace(0, np.nan)
    
    # Regime-Adaptive Signal Generation
    # High Volatility + High Volume Regime
    high_vol_volume_mask = (data['high_vol_regime'] == 1) & (data['high_volume_regime'] == 1)
    data['high_regime_signal'] = np.where(
        high_vol_volume_mask,
        data['price_momentum_5d'] * data['volume_intensity'] * (1 + data['momentum_acceleration']),
        0
    )
    
    # Low Volatility + Low Volume Regime
    low_vol_volume_mask = (data['low_vol_regime'] == 1) & (data['low_volume_regime'] == 1)
    data['low_regime_signal'] = np.where(
        low_vol_volume_mask,
        data['price_momentum_1d'] * data['amount_confirmed_momentum'] * data['trend_strength'],
        0
    )
    
    # Transition Regime (neither high nor low)
    transition_mask = (~high_vol_volume_mask) & (~low_vol_volume_mask)
    regime_strength_weight = data['vol_regime_strength'].clip(0.5, 1.5)
    data['transition_signal'] = np.where(
        transition_mask,
        (data['price_momentum_1d'] * 0.4 + data['price_momentum_5d'] * 0.6) * regime_strength_weight * data['timeframe_alignment'],
        0
    )
    
    # Combined Adaptive Factor
    data['adaptive_momentum_factor'] = (
        data['high_regime_signal'] + data['low_regime_signal'] + data['transition_signal']
    )
    
    # Normalize the factor
    factor = data['adaptive_momentum_factor'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    return factor
