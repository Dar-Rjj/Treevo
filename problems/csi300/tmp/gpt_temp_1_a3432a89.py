import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Market Microstructure factor with temporal decay patterns
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Multi-Scale Order Flow Imbalance
    # Trade-Level Pressure Accumulation
    data['prev_close'] = data['close'].shift(1)
    data['directional_volume'] = np.sign(data['close'] - data['prev_close']) * data['volume']
    
    # Cumulative pressure with exponential decay (rolling 1000 trades equivalent)
    lambda_decay = 0.995
    data['pressure_accumulation'] = data['directional_volume'].ewm(alpha=1-lambda_decay, adjust=False).mean()
    
    # Volume-Weighted Price Impact
    data['immediate_impact'] = ((data['high'] - data['low']) / (0.5 * (data['high'] + data['low']))) * data['volume']
    
    # Delayed impact correlation (using past correlation)
    data['returns'] = data['close'].pct_change()
    data['future_returns_5'] = data['returns'].shift(-5).rolling(5).sum()
    data['delayed_impact_corr'] = data['immediate_impact'].rolling(50).corr(data['future_returns_5'].shift(5))
    
    # Impact asymmetry
    up_trend = data['close'] > data['close'].rolling(5).mean()
    down_trend = data['close'] < data['close'].rolling(5).mean()
    impact_up = data.loc[up_trend, 'immediate_impact'].rolling(20).mean()
    impact_down = data.loc[down_trend, 'immediate_impact'].rolling(20).mean()
    data['impact_asymmetry'] = impact_up - impact_down
    
    # Microstructural Regime Classification
    # High-Frequency Noise Ratio
    data['ret_1min'] = data['close'].pct_change()
    data['ret_5min'] = data['close'].pct_change(5)
    data['noise_ratio'] = data['ret_1min'].rolling(30).std() / data['ret_5min'].rolling(30).std()
    data['noise_persistence'] = data['ret_1min'].rolling(30).apply(lambda x: x.autocorr(lag=1))
    
    # Liquidity Absorption Patterns
    data['volume_spread_ratio'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Regime classification
    momentum_accumulation = (data['noise_ratio'] < 0.7) & (data['volume_spread_ratio'] > data['volume_spread_ratio'].rolling(50).quantile(0.7))
    liquidity_exhaustion = (data['volume_spread_ratio'] > 1000) & (data['impact_asymmetry'] < -0.1)
    equilibrium_breakout = (data['pressure_accumulation'] < data['pressure_accumulation'].rolling(50).quantile(0.1)) & (data['delayed_impact_corr'] > 0.2)
    
    # Temporal Pattern Recognition
    # Intraday Seasonality (using time index if available)
    if hasattr(data.index, 'time'):
        data['hour_minute'] = data.index.time
        # Simple time-based seasonality proxy
        hour_seasonality = data.groupby(data.index.hour)['returns'].mean()
        data['time_effect'] = data.index.hour.map(hour_seasonality)
    else:
        data['time_effect'] = 0
    
    # Event-Driven Memory Effects
    data['volume_ma_20'] = data['volume'].rolling(20).mean()
    data['large_trade'] = data['volume'] > 5 * data['volume_ma_20']
    data['gap'] = abs(data['open'] - data['prev_close']) / data['prev_close']
    data['gap_event'] = data['gap'] > 0.02
    
    # Memory effects with different decay functions
    # Short-term memory (exponential decay, half-life = 3)
    data['short_term_memory'] = data['large_trade'].astype(float).ewm(halflife=3).mean()
    
    # Medium-term momentum (linear decay over 20 periods)
    weights = np.linspace(1, 0, 20)
    data['medium_term_momentum'] = data['returns'].rolling(20).apply(lambda x: np.dot(x, weights) if len(x) == 20 else np.nan)
    
    # Long-term reversal (hyperbolic decay proxy)
    data['long_term_reversal'] = -data['returns'].rolling(60).mean()
    
    # Cross-Asset Microstructure Spillovers (proxies)
    # Price leadership (auto-correlation based proxy)
    data['price_leadership'] = data['returns'].rolling(20).corr(data['returns'].shift(1))
    
    # Information diffusion (volatility-based proxy)
    data['volatility_20'] = data['returns'].rolling(20).std()
    data['information_diffusion'] = data['volatility_20'].pct_change(5)
    
    # Informed trading intensity
    data['large_trade_freq'] = data['large_trade'].rolling(20).mean()
    data['informed_trading_intensity'] = data['large_trade_freq'] * data['immediate_impact'].rolling(20).mean()
    
    # Adaptive Signal Generation
    # Momentum Accumulation Signal
    momentum_signal = momentum_accumulation & (data['pressure_accumulation'] > 2 * data['pressure_accumulation'].rolling(50).std())
    momentum_timing = (data['time_effect'] > 0) & (data['short_term_memory'] > 0.5)
    momentum_strength = data['pressure_accumulation'] * data['price_leadership'] * data['information_diffusion']
    
    # Liquidity Exhaustion Signal
    # Inventory accumulation proxy (using directional volume accumulation)
    data['inventory_accumulation'] = data['directional_volume'].cumsum()
    liquidity_signal = liquidity_exhaustion & (abs(data['inventory_accumulation']) > 3 * data['inventory_accumulation'].rolling(50).std())
    liquidity_timing = data['gap_event'] & (data['medium_term_momentum'] < -0.1)
    # Adverse selection proxy
    data['adverse_selection'] = ((data['high'] - data['low']) / data['close']) * (1 - data['short_term_memory'])
    liquidity_strength = abs(data['inventory_accumulation']) * data['adverse_selection'] * data['impact_asymmetry']
    
    # Equilibrium Breakout Signal
    # Flow divergence proxy (using pressure accumulation divergence)
    data['flow_divergence'] = data['pressure_accumulation'] - data['pressure_accumulation'].rolling(20).mean()
    equilibrium_signal = equilibrium_breakout & (data['flow_divergence'] > 0)
    equilibrium_timing = data['large_trade'] & (data['long_term_reversal'] > 0.2)
    equilibrium_strength = data['flow_divergence'] * data['informed_trading_intensity'] * data['delayed_impact_corr']
    
    # Combine signals with weights
    momentum_component = momentum_signal.astype(float) * momentum_timing.astype(float) * momentum_strength
    liquidity_component = liquidity_signal.astype(float) * liquidity_timing.astype(float) * liquidity_strength
    equilibrium_component = equilibrium_signal.astype(float) * equilibrium_timing.astype(float) * equilibrium_strength
    
    # Final factor combination
    factor = (
        0.4 * momentum_component.fillna(0) +
        0.35 * liquidity_component.fillna(0) +
        0.25 * equilibrium_component.fillna(0)
    )
    
    # Normalize and clean
    factor = (factor - factor.rolling(100).mean()) / factor.rolling(100).std()
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return factor
