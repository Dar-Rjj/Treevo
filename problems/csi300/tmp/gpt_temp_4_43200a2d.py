import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Volatility & Market Context
    # Volatility regime classification
    data['volatility_20d'] = data['returns'].rolling(window=20).std()
    data['volatility_60d_median'] = data['volatility_20d'].rolling(window=60).median()
    data['volatility_regime'] = np.where(data['volatility_20d'] > data['volatility_60d_median'], 'high', 'low')
    
    # Market environment assessment
    data['price_slope'] = (data['close'] - data['close'].shift(20)) / 20
    data['high_low_range_20d'] = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    data['atr_20d'] = (data['high'] - data['low']).rolling(window=20).mean()
    data['range_bound_indicator'] = data['high_low_range_20d'] / data['atr_20d']
    data['breakout_position'] = (data['close'] - data['low'].rolling(window=20).min()) / (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min())
    
    # Volume participation analysis
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_surge'] = data['volume'] > (2 * data['volume_20d_avg'])
    data['volume_change'] = data['volume'].pct_change()
    data['price_change'] = data['close'].pct_change()
    data['volume_leadership'] = data['volume_change'].rolling(window=5).corr(data['price_change'])
    data['volume_std_20d'] = data['volume'].rolling(window=20).std()
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(5)) / data['volume_std_20d']
    
    # Efficiency & Momentum Components
    # Price-volume efficiency metrics
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['return_10d'] = data['close'] / data['close'].shift(10) - 1
    data['volume_efficiency'] = data['return_10d'] / (data['volume'] * (data['high'] - data['low'])).replace(0, np.nan)
    data['closing_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-timeframe momentum signals
    data['vol_adj_momentum'] = data['return_10d'] / data['volatility_20d'].replace(0, np.nan)
    data['efficiency_5d_avg'] = data['intraday_efficiency'].rolling(window=5).mean()
    data['efficiency_momentum'] = (data['intraday_efficiency'] / data['efficiency_5d_avg'] - 1).replace([np.inf, -np.inf], np.nan)
    
    # Volume momentum persistence
    data['positive_volume_momentum'] = (data['volume_momentum'] > 0).astype(int)
    data['volume_momentum_persistence'] = data['positive_volume_momentum'].rolling(window=10).sum() / 10
    
    # Gap & breakout analysis
    data['gap_size'] = abs(data['open'] - data['close'].shift(1)) / data['atr_20d'].replace(0, np.nan)
    data['gap_fade_strength'] = (data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    
    # Support/resistance levels
    data['resistance_20d'] = data['high'].rolling(window=20).max()
    data['support_20d'] = data['low'].rolling(window=20).min()
    data['breakout_signal'] = (data['close'] - data['support_20d']) / (data['resistance_20d'] - data['support_20d']).replace(0, np.nan)
    
    # Liquidity & Shock Detection
    # Liquidity shock assessment
    data['volume_percentile_20d'] = data['volume'].rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)).astype(int))
    data['price_impact'] = (data['high'] - data['low']) / (data['volume'] * data['close']).replace(0, np.nan)
    
    # Trend exhaustion signals
    data['overbought'] = (data['close'] - data['high'].rolling(window=20).max()) / (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()).replace(0, np.nan)
    data['oversold'] = (data['low'].rolling(window=20).min() - data['close']) / (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()).replace(0, np.nan)
    
    # Momentum divergence
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['price_change_10d_avg'] = data['price_change'].rolling(window=10).mean()
    data['price_volume_divergence'] = ((data['volume'] - data['volume_10d_avg']) / 
                                     (data['price_change'] - data['price_change_10d_avg']).replace(0, np.nan))
    
    # Efficiency divergence
    data['efficiency_trend'] = data['intraday_efficiency'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data['efficiency_divergence'] = data['intraday_efficiency'] - data['efficiency_trend']
    
    # Regime-Adaptive Signal Processing
    # Volatility regime weighting
    high_vol_weight = np.where(data['volatility_regime'] == 'high', 1.2, 0.8)
    low_vol_weight = np.where(data['volatility_regime'] == 'low', 1.2, 0.8)
    
    # Market environment adaptation
    trend_regime = data['price_slope'].abs() > data['price_slope'].rolling(window=60).std()
    range_regime = data['range_bound_indicator'] < 1.0
    
    # Composite Alpha Generation
    # Core signal components
    efficiency_momentum = (data['intraday_efficiency'] * 0.3 + 
                          data['volume_efficiency'] * 0.4 + 
                          data['closing_pressure'] * 0.3)
    
    volume_momentum_component = (data['volume_leadership'] * 0.4 + 
                                data['volume_momentum'] * 0.3 + 
                                data['volume_momentum_persistence'] * 0.3)
    
    breakout_signals = (data['breakout_signal'] * 0.4 + 
                       data['gap_fade_strength'] * 0.3 + 
                       (1 - abs(data['overbought'])) * 0.3)
    
    # Context application with regime adjustments
    volatility_adjusted_momentum = data['vol_adj_momentum'] * high_vol_weight
    raw_momentum = data['return_10d'] * low_vol_weight
    
    # Apply market environment filters
    trend_adjusted = np.where(trend_regime, 
                             efficiency_momentum * 0.6 + volume_momentum_component * 0.4,
                             efficiency_momentum * 0.4 + volume_momentum_component * 0.6)
    
    range_adjusted = np.where(range_regime,
                             breakout_signals * 0.7 + efficiency_momentum * 0.3,
                             breakout_signals * 0.3 + efficiency_momentum * 0.7)
    
    # Final signal construction
    base_signal = (volatility_adjusted_momentum * 0.3 + 
                  raw_momentum * 0.2 + 
                  trend_adjusted * 0.25 + 
                  range_adjusted * 0.25)
    
    # Apply volume confirmation
    volume_confirmation = np.where(data['volume_surge'], 1.5, 1.0)
    volume_confirmation = np.where(data['volume'] < data['volume_20d_avg'] * 0.5, 0.5, volume_confirmation)
    
    # Apply divergence filters
    divergence_filter = np.where(data['price_volume_divergence'] > 0, 1.2, 0.8)
    divergence_filter = np.where(data['efficiency_divergence'] > 0, divergence_filter * 1.1, divergence_filter * 0.9)
    
    # Final composite alpha
    alpha_signal = base_signal * volume_confirmation * divergence_filter
    
    # Clean and return
    alpha_series = alpha_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
