import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Multi-scale Price-Volume Divergence Factor
    Combines price momentum, volume patterns, directional efficiency, and volatility context
    to detect divergence signals across multiple timeframes.
    """
    data = df.copy()
    
    # Calculate Multi-period Price Momentum
    data['price_momentum_5'] = data['close'].pct_change(5)
    data['price_momentum_10'] = data['close'].pct_change(10)
    data['price_momentum_21'] = data['close'].pct_change(21)
    
    # Compute Volume Momentum Patterns
    data['volume_roc_5'] = data['volume'].pct_change(5)
    data['volume_roc_10'] = data['volume'].pct_change(10)
    data['volume_roc_21'] = data['volume'].pct_change(21)
    
    # Volume concentration analysis
    data['intraday_range'] = data['high'] - data['low']
    data['up_move_volume'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['down_move_volume'] = np.where(data['close'] < data['open'], data['volume'], 0)
    
    # Rolling volume concentration metrics
    data['up_volume_ratio_5'] = data['up_move_volume'].rolling(5).sum() / data['volume'].rolling(5).sum()
    data['volume_skewness_5'] = (data['up_volume_ratio_5'] - 0.5) * 2
    
    # Assess Directional Efficiency
    data['upward_efficiency'] = (data['high'] - data['open']) / np.where(data['intraday_range'] == 0, 1, data['intraday_range'])
    data['downward_efficiency'] = (data['open'] - data['low']) / np.where(data['intraday_range'] == 0, 1, data['intraday_range'])
    
    # Efficiency persistence
    data['eff_corr_3'] = data['upward_efficiency'].rolling(3).corr(data['downward_efficiency'])
    
    # Efficiency trend using linear regression slope (3-day window)
    def efficiency_slope(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        return stats.linregress(x, series)[0]
    
    data['efficiency_trend'] = data['upward_efficiency'].rolling(3).apply(efficiency_slope, raw=False)
    
    # Detect Multi-scale Divergence Signals
    # Price-Volume Divergence Classification
    data['bullish_convergence'] = ((data['price_momentum_5'] > 0) & (data['volume_roc_5'] > 0)).astype(int)
    data['bearish_convergence'] = ((data['price_momentum_5'] < 0) & (data['volume_roc_5'] < 0)).astype(int)
    data['hidden_bullish'] = ((data['price_momentum_5'] > 0) & (data['volume_roc_5'] < 0)).astype(int)
    data['hidden_bearish'] = ((data['price_momentum_5'] < 0) & (data['volume_roc_5'] > 0)).astype(int)
    
    # Timeframe Divergence Analysis
    data['momentum_divergence_st_mt'] = data['price_momentum_5'] - data['price_momentum_10']
    data['price_volume_momentum_div'] = data['price_momentum_5'] - data['volume_roc_5']
    
    # Divergence Strength Measurement
    data['divergence_magnitude'] = (abs(data['momentum_divergence_st_mt']) + 
                                   abs(data['price_volume_momentum_div'])) / 2
    
    # Duration of consistent divergence patterns (5-day rolling)
    data['consistent_divergence'] = ((data['hidden_bullish'].rolling(5).sum() >= 3) | 
                                    (data['hidden_bearish'].rolling(5).sum() >= 3)).astype(int)
    
    # Incorporate Volatility Context
    data['upside_volatility'] = (data['high'] - data['close']) / data['close']
    data['downside_volatility'] = (data['close'] - data['low']) / data['close']
    data['total_volatility'] = data['upside_volatility'] + data['downside_volatility']
    
    # Volatility regime assessment (rolling percentiles)
    data['volatility_regime'] = data['total_volatility'].rolling(21).rank(pct=True)
    
    # Volatility-momentum interaction
    data['vol_expansion_momentum'] = data['total_volatility'] * data['price_momentum_5']
    
    # Volatility-adjusted divergence signals
    data['vol_adjusted_divergence'] = data['divergence_magnitude'] * data['volatility_regime']
    
    # Generate Alpha Factor
    # Combine multi-scale divergence components
    alpha_components = []
    
    # Weighted momentum divergence (short-term focus)
    momentum_div = (data['momentum_divergence_st_mt'] * 0.6 + 
                   data['price_volume_momentum_div'] * 0.4)
    
    # Divergence classification signals
    divergence_signals = (data['hidden_bullish'] - data['hidden_bearish'] + 
                         data['bullish_convergence'] * 0.5 - data['bearish_convergence'] * 0.5)
    
    # Efficiency-based adjustments
    efficiency_adj = data['efficiency_trend'] * data['eff_corr_3']
    
    # Volume confirmation
    volume_confirmation = data['volume_skewness_5'] * data['consistent_divergence']
    
    # Volatility context adjustment
    volatility_adj = np.where(data['volatility_regime'] > 0.7, 
                             data['vol_adjusted_divergence'] * 1.5,
                             data['vol_adjusted_divergence'])
    
    # Final alpha factor composition
    alpha_factor = (momentum_div * 0.3 +
                   divergence_signals * 0.25 +
                   efficiency_adj * 0.2 +
                   volume_confirmation * 0.15 +
                   volatility_adj * 0.1)
    
    # Clean and return the factor
    alpha_series = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
