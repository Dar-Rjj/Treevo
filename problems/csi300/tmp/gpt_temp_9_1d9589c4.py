import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum with Liquidity Confirmation
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # Intraday Range Efficiency
    data['intraday_range_eff'] = (data['high'] - data['low']).abs() / (data['close'].shift(1) - data['open'].shift(1)).abs()
    data['intraday_range_eff'] = data['intraday_range_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Gap Volatility Impact
    data['gap_vol_impact'] = (data['open'] - data['close'].shift(1)).abs() / (data['high'].shift(1) - data['low'].shift(1)).abs()
    data['gap_vol_impact'] = data['gap_vol_impact'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility Persistence (5-day rolling)
    data['range_expansion'] = (data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1))
    data['vol_persistence'] = data['range_expansion'].rolling(window=5, min_periods=3).mean()
    
    # Volatility Regime Score
    data['vol_regime_score'] = (
        data['intraday_range_eff'].rolling(window=5, min_periods=3).mean() +
        data['gap_vol_impact'].rolling(window=5, min_periods=3).mean() +
        data['vol_persistence']
    ) / 3
    
    # Classify regimes
    vol_quantiles = data['vol_regime_score'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.qcut(x, q=[0, 0.3, 0.7, 1.0], labels=False, duplicates='drop').iloc[-1] 
        if len(x.dropna()) >= 10 else np.nan, raw=False
    )
    
    data['vol_regime'] = np.where(vol_quantiles == 2, 'high', 
                                 np.where(vol_quantiles == 0, 'low', 'transition'))
    
    # Liquidity Flow Dynamics
    # Volume-Price Efficiency
    data['volume_price_eff'] = data['volume'] / (data['close'] - data['close'].shift(1)).abs()
    data['volume_price_eff'] = data['volume_price_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Trade Intensity
    data['trade_intensity'] = data['amount'] / data['volume']
    data['trade_intensity'] = data['trade_intensity'].replace([np.inf, -np.inf], np.nan)
    
    # Flow Concentration
    data['flow_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean().shift(1)
    
    # Liquidity Flow Score
    data['liquidity_flow_score'] = (
        data['volume_price_eff'].rolling(window=5, min_periods=3).mean() +
        data['trade_intensity'].rolling(window=5, min_periods=3).mean() +
        data['flow_concentration']
    ) / 3
    
    # Regime-Adaptive Momentum
    # Volatility-Weighted Return (5-day)
    vol_range_5d = (data['high'].rolling(window=5, min_periods=3).max() - 
                   data['low'].rolling(window=5, min_periods=3).min())
    data['vol_weighted_return'] = ((data['close'] / data['close'].shift(5) - 1) / 
                                  vol_range_5d.replace(0, np.nan))
    
    # Liquidity-Confirmed Momentum
    data['liquidity_confirmed_momentum'] = (
        np.sign(data['close'] - data['close'].shift(1)) * 
        data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean().shift(1)
    )
    
    # Regime Stability Score
    regime_stability = data['vol_regime_score'].rolling(window=10, min_periods=5).std()
    data['regime_stability_score'] = 1 / (1 + regime_stability)
    
    # Regime-Stable Momentum (10-day)
    data['regime_stable_momentum'] = (
        (data['close'] / data['close'].shift(10) - 1) * 
        data['regime_stability_score']
    )
    
    # Cross-Factor Integration
    # High Volatility: Momentum * Volume Efficiency
    high_vol_factor = data['vol_weighted_return'] * data['volume_price_eff']
    
    # Low Volatility: Momentum * Trade Intensity  
    low_vol_factor = data['regime_stable_momentum'] * data['trade_intensity']
    
    # Transition Regimes: Momentum * Flow Concentration
    transition_factor = data['liquidity_confirmed_momentum'] * data['flow_concentration']
    
    # Regime-adaptive selection
    data['regime_adaptive_momentum'] = np.where(
        data['vol_regime'] == 'high', high_vol_factor,
        np.where(data['vol_regime'] == 'low', low_vol_factor, transition_factor)
    )
    
    # Final Alpha Factors
    # Primary: Volatility regime * Liquidity flow * Momentum strength
    primary_factor = (
        data['vol_regime_score'] * 
        data['liquidity_flow_score'] * 
        data['regime_adaptive_momentum']
    )
    
    # Secondary: Regime-adaptive momentum efficiency score
    momentum_efficiency = (
        data['regime_adaptive_momentum'].rolling(window=10, min_periods=5).mean() /
        data['regime_adaptive_momentum'].rolling(window=10, min_periods=5).std()
    )
    secondary_factor = momentum_efficiency * data['regime_stability_score']
    
    # Composite: Multi-scale volatility-liquidity momentum alignment
    short_term_momentum = data['liquidity_confirmed_momentum'].rolling(window=5, min_periods=3).mean()
    medium_term_momentum = data['vol_weighted_return'].rolling(window=10, min_periods=5).mean()
    long_term_momentum = data['regime_stable_momentum'].rolling(window=20, min_periods=10).mean()
    
    composite_factor = (
        short_term_momentum * 0.3 + 
        medium_term_momentum * 0.4 + 
        long_term_momentum * 0.3
    ) * data['liquidity_flow_score']
    
    # Final alpha factor (weighted combination)
    alpha_factor = (
        primary_factor.rank(pct=True) * 0.5 +
        secondary_factor.rank(pct=True) * 0.3 + 
        composite_factor.rank(pct=True) * 0.2
    )
    
    return alpha_factor
