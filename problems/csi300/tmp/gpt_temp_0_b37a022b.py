import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Dynamic Liquidity Absorption Factor
    df = df.copy()
    
    # Bidirectional Price Impact
    df['upside_absorption'] = (df['high'] - df['open']) * df['volume']
    df['downside_absorption'] = (df['open'] - df['low']) * df['volume']
    
    # Absorption Ratio
    df['net_absorption'] = df['upside_absorption'] - df['downside_absorption']
    df['total_absorption'] = df['upside_absorption'] + df['downside_absorption']
    df['absorption_ratio'] = df['net_absorption'] / df['total_absorption']
    df['absorption_ratio'] = df['absorption_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Price Trend (5-day slope)
    def calc_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    df['price_trend'] = df['close'].rolling(window=5, min_periods=5).apply(calc_slope, raw=False)
    
    # Final Signal
    df['liquidity_factor'] = df['absorption_ratio'] * df['price_trend']
    
    # Volatility Clustering Breakout Detector
    # Volatility Regimes
    df['short_vol'] = (df['high'] - df['low']).rolling(window=5, min_periods=5).mean()
    df['medium_vol'] = (df['high'] - df['low']).rolling(window=20, min_periods=20).mean()
    
    # Volatility Breakout
    df['vol_ratio'] = df['short_vol'] / df['medium_vol']
    df['bar_strength'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['bar_strength'] = df['bar_strength'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=10, min_periods=10).mean()
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Breakout Signal
    df['raw_breakout'] = df['vol_ratio'] * df['bar_strength'] * df['volume_ratio']
    df['directional_bias'] = np.sign(df['close'] - df['open'])
    df['volatility_factor'] = df['raw_breakout'] * df['directional_bias']
    
    # Momentum Divergence Oscillator
    # Price Momentum
    df['fast_momentum'] = df['close'] / df['close'].shift(2) - 1
    df['slow_momentum'] = df['close'] / df['close'].shift(9) - 1
    
    # Momentum Divergence
    df['momentum_spread'] = df['fast_momentum'] - df['slow_momentum']
    df['direction_indicator'] = np.sign(df['fast_momentum'] * df['slow_momentum'])
    
    # Volume Trend (5-day slope)
    df['volume_trend'] = df['volume'].rolling(window=5, min_periods=5).apply(calc_slope, raw=False)
    
    # Divergence Signal
    df['momentum_factor'] = df['momentum_spread'] * df['direction_indicator'] * df['volume_trend']
    
    # Price-Volume Congestion Breakout
    # ATR calculations
    def calc_atr(high, low, close, window):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window, min_periods=window).mean()
    
    df['atr_10'] = calc_atr(df['high'], df['low'], df['close'], 10)
    df['atr_20'] = calc_atr(df['high'], df['low'], df['close'], 20)
    
    # Trading Range Congestion
    df['price_compression'] = df['atr_10'] / df['atr_20']
    df['price_compression'] = df['price_compression'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    df['volume_drying'] = df['volume'] / df['volume'].rolling(window=10, min_periods=10).max()
    df['volume_drying'] = df['volume_drying'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Breakout Initiation
    df['price_breakout'] = (df['high'] - df['low']) / df['atr_10']
    df['price_breakout'] = df['price_breakout'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df['volume_expansion'] = df['volume'] / df['volume'].rolling(window=10, min_periods=10).mean()
    df['volume_expansion'] = df['volume_expansion'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Congestion Signal
    df['breakout_strength'] = df['price_breakout'] * df['volume_expansion']
    df['directional_weight'] = 2 * (df['close'] - df['low']) / (df['high'] - df['low']) - 1
    df['directional_weight'] = df['directional_weight'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df['congestion_factor'] = df['breakout_strength'] * df['directional_weight']
    
    # Combine all factors with equal weights
    factors = ['liquidity_factor', 'volatility_factor', 'momentum_factor', 'congestion_factor']
    df['composite_factor'] = df[factors].mean(axis=1)
    
    return df['composite_factor']
