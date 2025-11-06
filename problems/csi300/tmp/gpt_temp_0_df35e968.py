import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Volatility Regime Classification
    # ATR calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_5'] = data['tr'].rolling(window=5).mean()
    
    # Medium-term volatility
    data['vol_20'] = data['returns'].rolling(window=20).std()
    
    # Volatility ratio regime
    data['vol_ratio'] = data['atr_5'] / data['vol_20']
    data['vol_regime'] = np.where(data['vol_ratio'] > 1.2, 2, 
                                 np.where(data['vol_ratio'] < 0.8, 0, 1))
    
    # 2. Regime-Adaptive Momentum
    data['momentum_10'] = data['close'].pct_change(10)
    
    # Regime-weighted momentum adjustment
    regime_weights = {0: 0.6, 1: 1.0, 2: 1.4}  # Low, Normal, High volatility
    data['regime_momentum'] = data['momentum_10'] * data['vol_regime'].map(regime_weights)
    
    # 3. Volume-Volatility Coupling
    # Volume/True Range correlation (5-day rolling)
    data['volume_tr_corr'] = data['volume'].rolling(window=5).corr(data['tr'])
    
    # Volume shock detection
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    data['volume_shock'] = (data['volume'] - data['volume_ma_20']) / data['volume_ma_20']
    
    # 4. Directional Impact Asymmetry
    # Up-volume vs down-volume analysis
    data['price_change'] = data['close'] - data['open']
    data['up_volume'] = np.where(data['price_change'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['price_change'] < 0, data['volume'], 0)
    
    data['up_volume_ratio'] = data['up_volume'].rolling(window=5).sum() / \
                             (data['up_volume'].rolling(window=5).sum() + 
                              data['down_volume'].rolling(window=5).sum() + 1e-8)
    
    # Price-volume co-movement divergence
    data['price_volume_corr'] = data['returns'].rolling(window=10).corr(data['volume'].pct_change())
    
    # 5. Multi-scale Efficiency Measurement
    # Fractal dimension approximation using Hurst exponent method
    def hurst_exponent(series, window):
        lags = range(2, 6)
        tau = []
        for lag in lags:
            ts = series.rolling(window=lag).std()
            tau.append(np.log(ts.dropna().iloc[-1] if len(ts.dropna()) > 0 else 1.0))
        if len(tau) > 1:
            poly = np.polyfit(np.log(lags), tau, 1)
            return poly[0]
        return 0.5
    
    # Short-term efficiency (5-day)
    data['eff_short'] = data['close'].rolling(window=20).apply(
        lambda x: hurst_exponent(x, 5) if len(x.dropna()) >= 5 else 0.5, raw=False
    )
    
    # Medium-term efficiency (20-day)
    data['eff_medium'] = data['close'].rolling(window=40).apply(
        lambda x: hurst_exponent(x, 20) if len(x.dropna()) >= 20 else 0.5, raw=False
    )
    
    # 6. Breakout Quality Scoring
    # Recent high/low breakout detection
    data['high_20'] = data['high'].rolling(window=20).max()
    data['low_20'] = data['low'].rolling(window=20).min()
    
    data['breakout_high'] = (data['close'] > data['high_20'].shift(1)).astype(int)
    data['breakout_low'] = (data['close'] < data['low_20'].shift(1)).astype(int)
    
    # Volume confirmation
    data['volume_confirmation'] = np.where(
        (data['breakout_high'] == 1) | (data['breakout_low'] == 1),
        data['volume_shock'],
        0
    )
    
    # Volatility expansion
    data['vol_expansion'] = data['tr'] / data['tr'].rolling(window=5).mean()
    
    # Efficiency-based filtering
    data['efficiency_filter'] = np.where(
        (data['eff_short'] < 0.4) & (data['eff_medium'] < 0.4),  # Low efficiency = trending
        1.0, 0.5
    )
    
    data['breakout_score'] = (
        ((data['breakout_high'] - data['breakout_low']) * 
         data['volume_confirmation'] * 
         data['vol_expansion'] * 
         data['efficiency_filter'])
    ).rolling(window=3).mean()
    
    # 7. Liquidity Regime Identification
    # Volume-price elasticity
    data['price_volume_elasticity'] = (
        data['returns'].rolling(window=10).std() / 
        (data['volume'].pct_change().rolling(window=10).std() + 1e-8)
    )
    
    data['liquidity_regime'] = np.where(
        data['price_volume_elasticity'] > data['price_volume_elasticity'].rolling(window=20).quantile(0.7),
        2,  # Low liquidity
        np.where(data['price_volume_elasticity'] < data['price_volume_elasticity'].rolling(window=20).quantile(0.3),
                0,  # High liquidity
                1)  # Normal liquidity
    )
    
    # 8. Multi-Component Synthesis
    liquidity_weights = {0: 1.2, 1: 1.0, 2: 0.8}  # High, Normal, Low liquidity
    
    # Final factor synthesis
    data['final_factor'] = (
        data['regime_momentum'] * 0.4 +
        data['up_volume_ratio'] * 0.2 +
        data['price_volume_corr'] * 0.15 +
        data['breakout_score'] * 0.25
    ) * data['liquidity_regime'].map(liquidity_weights)
    
    # Clean up and return
    result = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
