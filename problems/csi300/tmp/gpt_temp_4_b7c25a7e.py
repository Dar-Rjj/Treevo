import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Asymmetry with Regime-Switching Efficiency factor
    """
    data = df.copy()
    
    # Price-Momentum Regime Detection
    # Calculate returns and momentum persistence
    data['returns'] = data['close'].pct_change()
    data['return_sign'] = np.sign(data['returns'])
    
    # Momentum persistence (3-day rolling)
    data['momentum_persistence'] = data['return_sign'].rolling(window=3).apply(
        lambda x: len(set(x)) if len(x) == 3 else np.nan, raw=True
    )
    data['momentum_strength'] = data['returns'].rolling(window=3).std()
    
    # Volume Asymmetry Analysis (5-day rolling)
    up_days = data['returns'] > 0
    down_days = data['returns'] < 0
    
    data['up_volume'] = data['volume'].where(up_days, 0)
    data['down_volume'] = data['volume'].where(down_days, 0)
    
    data['up_volume_5d'] = data['up_volume'].rolling(window=5).sum()
    data['down_volume_5d'] = data['down_volume'].rolling(window=5).sum()
    data['volume_asymmetry'] = (data['up_volume_5d'] - data['down_volume_5d']) / (data['up_volume_5d'] + data['down_volume_5d'])
    
    # Regime classification
    data['momentum_regime'] = np.where(
        (data['momentum_persistence'] == 1) & (data['momentum_strength'] > data['momentum_strength'].rolling(20).mean()),
        'strong', 'weak'
    )
    
    # Efficiency Gap with Microstructure Signals
    # Daily range efficiency
    data['daily_range'] = data['high'] - data['low']
    data['price_change'] = abs(data['close'] - data['open'])
    data['range_efficiency'] = data['daily_range'] / (data['price_change'] + 1e-8)
    
    # Trade size distribution
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_vol'] = data['avg_trade_size'].rolling(window=5).std()
    
    # Price impact asymmetry
    data['price_impact_buy'] = np.where(
        data['close'] > data['open'],
        (data['close'] - data['open']) / data['open'],
        0
    )
    data['price_impact_sell'] = np.where(
        data['close'] < data['open'],
        (data['open'] - data['close']) / data['open'],
        0
    )
    data['impact_asymmetry'] = data['price_impact_buy'] - data['price_impact_sell']
    
    # Volatility Clustering with Momentum Alignment
    data['volatility_5d'] = data['daily_range'].rolling(window=5).std()
    data['volatility_regime'] = np.where(
        data['volatility_5d'] > data['volatility_5d'].rolling(20).mean(),
        'high', 'low'
    )
    
    # Volatility-adjusted momentum
    data['vol_adj_momentum'] = data['returns'].rolling(window=3).mean() / (data['volatility_5d'] + 1e-8)
    
    # Volume-volatility interaction
    data['volume_persistence'] = data['volume'].rolling(window=5).apply(
        lambda x: x.pct_change().std(), raw=False
    )
    
    # Liquidity Flow with Price Discovery Efficiency
    # Liquidity momentum
    data['liquidity_momentum'] = data['avg_trade_size'].pct_change(periods=3)
    
    # Price discovery efficiency
    data['prev_close'] = data['close'].shift(1)
    data['opening_efficiency'] = abs(data['open'] - data['prev_close']) / (data['daily_range'] + 1e-8)
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['closing_efficiency'] = abs(data['close'] - data['midpoint']) / (data['daily_range'] + 1e-8)
    
    # Flow-price divergence
    data['flow_momentum_alignment'] = data['liquidity_momentum'] * data['returns'].rolling(window=3).mean()
    
    # Multi-Timeframe Signal Integration
    # Short-term vs medium-term momentum
    data['momentum_3d'] = data['close'].pct_change(periods=3)
    data['momentum_8d'] = data['close'].pct_change(periods=8)
    data['timeframe_convergence'] = np.sign(data['momentum_3d']) * np.sign(data['momentum_8d'])
    
    # Volume confirmation across timeframes
    data['volume_3d'] = data['volume'].rolling(window=3).mean()
    data['volume_8d'] = data['volume'].rolling(window=8).mean()
    data['volume_persistence_ratio'] = data['volume_3d'] / (data['volume_8d'] + 1e-8)
    
    # Regime-dependent weighting
    regime_weights = {
        'strong': 1.2,
        'weak': 0.8,
        'high': 0.9,
        'low': 1.1
    }
    
    # Final signal generation with regime-aware weighting
    data['momentum_component'] = (
        data['momentum_3d'] * data['timeframe_convergence'] * 
        np.where(data['momentum_regime'] == 'strong', regime_weights['strong'], regime_weights['weak'])
    )
    
    data['volume_component'] = (
        data['volume_asymmetry'] * data['volume_persistence_ratio'] *
        np.where(data['volatility_regime'] == 'high', regime_weights['high'], regime_weights['low'])
    )
    
    data['efficiency_component'] = (
        data['range_efficiency'] * data['flow_momentum_alignment'] *
        data['impact_asymmetry']
    )
    
    # Risk-Adjusted Factor Output
    # Volatility regime adjustment
    volatility_scaling = np.where(
        data['volatility_regime'] == 'high',
        1 / (data['volatility_5d'] + 1e-8),
        1.0
    )
    
    # Liquidity quality check
    liquidity_filter = np.where(
        data['avg_trade_size'] > data['avg_trade_size'].rolling(20).quantile(0.1),
        1.0, 0.5
    )
    
    # Final alpha factor
    data['alpha_factor'] = (
        data['momentum_component'] * 0.4 +
        data['volume_component'] * 0.35 +
        data['efficiency_component'] * 0.25
    ) * volatility_scaling * liquidity_filter
    
    # Clean up and return
    result = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
