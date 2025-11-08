import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor combining intraday regime momentum, smart flow reversal detection,
    range breakout quality, and liquidity-weighted mean reversion signals.
    """
    data = df.copy()
    
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # 1. Intraday Regime Momentum Factor
    # Opening gap momentum
    data['gap_momentum'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_strength'] = abs(data['gap_momentum'])
    
    # Intraday range efficiency
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Closing momentum
    data['closing_momentum'] = (data['close'] - data['midpoint']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volatility regime
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['vol_5d_avg'] = data['daily_range'].rolling(window=5).mean()
    data['vol_20d_median'] = data['daily_range'].rolling(window=20).median()
    data['vol_regime'] = np.where(data['vol_5d_avg'] > data['vol_20d_median'], 1.2, 0.8)
    
    # Volume regime
    data['vol_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['volume_ratio'] = data['volume'] / data['vol_10d_avg']
    data['volume_trend'] = data['volume'].rolling(window=3).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    data['volume_regime'] = np.where(data['volume_ratio'] > 1.2, 1.1, 0.9)
    
    # Combine intraday momentum components
    data['intraday_momentum'] = (
        data['gap_strength'] * np.sign(data['gap_momentum']) * 0.3 +
        data['range_efficiency'] * 0.4 +
        data['closing_momentum'] * 0.3
    )
    
    # Regime-adjusted momentum
    data['regime_momentum'] = data['intraday_momentum'] * data['vol_regime'] * data['volume_regime']
    
    # 2. Smart Flow Reversal Detection Factor
    # Order flow imbalance
    data['bullish_flow'] = np.where(data['close'] > data['open'], data['amount'], 0)
    data['bearish_flow'] = np.where(data['close'] < data['open'], data['amount'], 0)
    data['net_flow_ratio'] = (data['bullish_flow'] - data['bearish_flow']) / data['amount'].replace(0, np.nan)
    data['net_flow_3d'] = data['net_flow_ratio'].rolling(window=3).mean()
    
    # Flow persistence
    data['flow_direction'] = np.sign(data['net_flow_ratio'])
    data['flow_persistence'] = data['flow_direction'].rolling(window=5).apply(
        lambda x: len(set(x)) == 1 and x.notna().all(), raw=False
    ).astype(float)
    
    # Price-flow divergence
    data['price_change'] = data['close'].pct_change(3)
    data['flow_price_divergence'] = np.where(
        np.sign(data['net_flow_3d']) != np.sign(data['price_change']),
        abs(data['net_flow_3d']) * abs(data['price_change']),
        0
    )
    
    # Flow reversal signal
    data['flow_reversal'] = data['flow_price_divergence'] * data['flow_persistence'] * -1
    
    # 3. Range Breakout Quality Score
    # Price range expansion
    data['true_range_10d_avg'] = data['true_range'].rolling(window=10).mean()
    data['range_expansion'] = data['true_range'] / data['true_range_10d_avg']
    
    # Support/resistance break
    data['20d_high'] = data['high'].rolling(window=20).max()
    data['20d_low'] = data['low'].rolling(window=20).min()
    data['breakout_strength'] = np.where(
        data['close'] > data['20d_high'],
        (data['close'] - data['20d_high']) / data['20d_high'],
        np.where(
            data['close'] < data['20d_low'],
            (data['close'] - data['20d_low']) / data['20d_low'],
            0
        )
    )
    
    # Volume breakout confirmation
    data['vol_15d_avg'] = data['volume'].rolling(window=15).mean()
    data['volume_breakout'] = data['volume'] / data['vol_15d_avg']
    
    # Breakout quality
    data['breakout_quality'] = (
        data['range_expansion'] * 0.4 +
        abs(data['breakout_strength']) * 0.4 +
        data['volume_breakout'] * 0.2
    ) * np.sign(data['breakout_strength'])
    
    # 4. Liquidity-Weighted Mean Reversion
    # Price extremity
    data['ma_5d'] = data['close'].rolling(window=5).mean()
    data['price_deviation'] = (data['close'] - data['ma_5d']) / data['ma_5d']
    data['10d_range'] = data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()
    data['range_position'] = (data['close'] - data['low'].rolling(window=10).min()) / data['10d_range'].replace(0, np.nan)
    data['extremity_score'] = abs(data['price_deviation']) * (1 - 2 * abs(data['range_position'] - 0.5))
    
    # Liquidity conditions
    data['effective_spread'] = (data['high'] - data['low']) / data['midpoint']
    data['spread_10d_avg'] = data['effective_spread'].rolling(window=10).mean()
    data['spread_regime'] = np.where(data['effective_spread'] > data['spread_10d_avg'], 0.8, 1.2)
    
    data['vol_8d_avg'] = data['volume'].rolling(window=8).mean()
    data['volume_concentration'] = data['volume'] / data['vol_8d_avg']
    
    # Liquidity composite
    data['liquidity_score'] = data['spread_regime'] * data['volume_concentration']
    
    # Mean reversion signal
    data['mean_reversion'] = -data['price_deviation'] * data['extremity_score'] * data['liquidity_score']
    
    # Combine all factors with dynamic weights
    momentum_weight = 0.25
    reversal_weight = 0.25
    breakout_weight = 0.25
    mean_reversion_weight = 0.25
    
    # Calculate final composite factor
    factor = (
        data['regime_momentum'] * momentum_weight +
        data['flow_reversal'] * reversal_weight +
        data['breakout_quality'] * breakout_weight +
        data['mean_reversion'] * mean_reversion_weight
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor
