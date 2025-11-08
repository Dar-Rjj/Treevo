import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a composite alpha factor combining volume-weighted momentum, 
    liquidity efficiency, order flow imbalance, volatility regime breakout, 
    and price-range efficiency signals.
    """
    df = data.copy()
    
    # Volume-Weighted Price Momentum Regime Factor
    # Multi-timeframe price ratios
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)
    
    # Exponential smoothing
    df['mom_short'] = df['ret_1'].ewm(span=3).mean()
    df['mom_medium'] = df['ret_5'].ewm(span=8).mean()
    df['mom_long'] = df['ret_20'].ewm(span=15).mean()
    
    # Volume persistence
    df['vol_ratio_1'] = df['volume'] / df['volume'].shift(1)
    df['vol_ratio_5'] = df['volume'] / df['volume'].shift(5)
    df['vol_ratio_20'] = df['volume'] / df['volume'].shift(20)
    
    # Volume-weighted momentum
    df['vw_mom_short'] = df['mom_short'] * df['vol_ratio_1']
    df['vw_mom_medium'] = df['mom_medium'] * df['vol_ratio_5']
    df['vw_mom_long'] = df['mom_long'] * df['vol_ratio_20']
    
    # Volatility-adjusted weighting
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()
    
    # Momentum factor
    momentum_factor = (
        df['vw_mom_short'] * (1 / (1 + df['vol_5'])) +
        df['vw_mom_medium'] * (1 / (1 + df['vol_20'])) +
        df['vw_mom_long'] * (1 / (1 + df['vol_20']))
    ) / 3
    
    # Liquidity-Efficiency Price Impact Factor
    # Price impact efficiency
    df['price_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['typical_range'] = df['price_range'].rolling(5).median()
    df['range_efficiency'] = df['price_range'] / df['typical_range']
    
    # Volume efficiency
    df['volume_efficiency'] = df['amount'] / (df['high'] - df['low'])
    df['hist_volume_eff'] = df['volume_efficiency'].rolling(20).median()
    df['volume_eff_ratio'] = df['volume_efficiency'] / df['hist_volume_eff']
    
    # Liquidity conditions
    df['vol_concentration_1'] = df['volume'] / df['volume'].shift(1)
    df['vol_concentration_5'] = df['volume'] / df['volume'].rolling(5).mean()
    df['concentration_persistence'] = df['vol_concentration_1'] * df['vol_concentration_5']
    
    # Bid-ask spread proxy
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['spread_proxy'] = (df['high'] - df['low']) / df['typical_price']
    df['avg_spread'] = df['spread_proxy'].rolling(5).mean()
    df['spread_ratio'] = df['spread_proxy'] / df['avg_spread']
    
    # Liquidity factor
    liquidity_factor = (
        df['range_efficiency'] * df['volume_eff_ratio'] * 
        (1 / (1 + df['spread_ratio'])) * df['concentration_persistence']
    )
    
    # Order Flow Imbalance Persistence Factor
    # Directional amount flow
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] < df['open']).astype(int)
    df['net_amount'] = df['up_day'] * df['amount'] - df['down_day'] * df['amount']
    df['net_amount_5'] = df['net_amount'].rolling(5).sum()
    df['net_amount_20'] = df['net_amount'].rolling(20).sum()
    
    # Order flow persistence
    df['amount_streak'] = 0
    streak = 0
    for i in range(1, len(df)):
        if df['net_amount'].iloc[i] * df['net_amount'].iloc[i-1] > 0:
            streak += 1
        else:
            streak = 1
        df.iloc[i, df.columns.get_loc('amount_streak')] = streak
    
    df['flow_intensity'] = df['net_amount'] / df['amount'].rolling(5).mean()
    df['flow_momentum'] = df['amount_streak'] * df['flow_intensity']
    
    # Order flow factor
    order_flow_factor = df['net_amount_5'] * df['amount_streak'] * df['flow_momentum']
    
    # Volatility-Regime Breakout Confidence Factor
    # Volatility regime detection
    df['vol_5_range'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']
    df['vol_10_range'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close']
    df['vol_20_range'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
    
    df['vol_regime_strength'] = (
        df['vol_5_range'] / df['vol_5_range'].rolling(20).median() +
        df['vol_10_range'] / df['vol_10_range'].rolling(20).median() +
        df['vol_20_range'] / df['vol_20_range'].rolling(20).median()
    ) / 3
    
    # Breakout quality
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['true_range'].rolling(14).mean()
    df['breakout_quality'] = df['true_range'] / df['atr']
    
    # Volume confirmation
    df['breakout_volume'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_range_corr'] = df['volume'].rolling(5).corr(df['true_range'])
    
    # Volatility factor
    volatility_factor = (
        df['vol_regime_strength'] * df['breakout_quality'] * 
        df['breakout_volume'] * (1 + df['volume_range_corr'])
    )
    
    # Price-Range Efficiency Momentum Factor
    # Range-based momentum
    df['range_efficiency_day'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['range_eff_persistence'] = 0
    eff_streak = 0
    for i in range(1, len(df)):
        if df['range_efficiency_day'].iloc[i] > df['range_efficiency_day'].iloc[i-1]:
            eff_streak += 1
        else:
            eff_streak = max(0, eff_streak - 1)
        df.iloc[i, df.columns.get_loc('range_eff_persistence')] = eff_streak
    
    # Volume-range relationship
    df['volume_per_range'] = df['amount'] / (df['high'] - df['low'])
    df['volume_eff_trend'] = df['volume_per_range'] / df['volume_per_range'].rolling(5).mean()
    
    # Efficiency factor
    efficiency_factor = (
        df['range_efficiency_day'] * df['volume_eff_trend'] * 
        (1 + df['range_eff_persistence'] / 10)
    )
    
    # Composite factor
    composite_factor = (
        momentum_factor.rank(pct=True) * 0.25 +
        liquidity_factor.rank(pct=True) * 0.20 +
        order_flow_factor.rank(pct=True) * 0.20 +
        volatility_factor.rank(pct=True) * 0.20 +
        efficiency_factor.rank(pct=True) * 0.15
    )
    
    return composite_factor
