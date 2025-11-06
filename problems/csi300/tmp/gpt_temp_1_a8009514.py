import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor combining multiple technical signals:
    - Volatility Regime Adjusted Price Momentum
    - Volume-Confirmed Breakout Strength  
    - Intraday Momentum Persistence
    - Liquidity-Adjusted Reversal Factor
    - Multi-Timeframe Volume Divergence
    - Volatility-Volume Compression Expansion
    - Price Gap Mean Reversion Strength
    - Momentum Acceleration Deceleration
    """
    
    df = data.copy()
    
    # Volatility Regime Adjusted Price Momentum
    df['short_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['long_momentum'] = df['close'] / df['close'].shift(21) - 1
    
    # Recent volatility (20 days)
    df['recent_vol'] = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close'].rolling(window=20).mean()
    
    # Historical volatility (252 days)
    df['hist_vol'] = (df['high'].rolling(window=252).max() - df['low'].rolling(window=252).min()) / df['close'].rolling(window=252).mean()
    
    # Volatility regime
    df['vol_regime'] = df['recent_vol'] / df['hist_vol']
    
    # Regime-adjusted momentum
    df['regime_adj_momentum'] = np.where(
        df['vol_regime'] > 1,  # High volatility regime
        df['short_momentum'],  # Prefer short-term momentum
        df['long_momentum']    # Prefer long-term momentum
    )
    
    # Volume-Confirmed Breakout Strength
    df['resistance'] = df['high'].shift(1).rolling(window=20).max()
    df['breakout'] = (df['high'] > df['resistance']).astype(int)
    df['avg_volume'] = df['volume'].shift(1).rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['avg_volume']
    df['breakout_strength'] = df['breakout'] * (df['high'] - df['resistance']) / df['resistance'] * df['volume_ratio']
    
    # Intraday Momentum Persistence
    # Assuming we have intraday data - using open to close as proxy
    df['morning_momentum'] = (df['close'] - df['open']) / df['open']
    df['afternoon_momentum'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)  # Simplified
    df['momentum_continuity'] = np.sign(df['morning_momentum']) * np.sign(df['afternoon_momentum'])
    df['persistence_signal'] = df['momentum_continuity'] * (abs(df['morning_momentum']) + abs(df['afternoon_momentum']))
    
    # Liquidity-Adjusted Reversal Factor
    df['price_reversal'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['liquidity'] = df['amount'] / df['volume']  # Average price as liquidity proxy
    df['liquidity_adj_reversal'] = df['price_reversal'] * df['liquidity'] / df['liquidity'].rolling(window=20).mean()
    
    # Multi-Timeframe Volume Divergence
    df['short_volume_trend'] = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=True)
    df['medium_volume_trend'] = df['volume'].rolling(window=20).apply(lambda x: np.polyfit(range(20), x, 1)[0], raw=True)
    df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
    df['volume_divergence'] = np.where(
        df['short_volume_trend'] * df['medium_volume_trend'] < 0,  # Conflicting trends
        abs(df['short_volume_trend'] - df['medium_volume_trend']) * (1 - df['price_volume_corr']),
        0
    )
    
    # Volatility-Volume Compression Expansion
    df['volatility_range'] = (df['high'] - df['low']) / df['close']
    df['vol_compression'] = df['volatility_range'].rolling(window=10).std()
    df['volume_surge'] = df['volume'] / df['volume'].rolling(window=10).mean()
    df['compression_break'] = (df['vol_compression'] < df['vol_compression'].rolling(window=20).quantile(0.3)) & (df['volume_surge'] > 1.5)
    df['pattern_signal'] = df['compression_break'].astype(int) * df['volume_surge'] * df['volatility_range']
    
    # Price Gap Mean Reversion Strength
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_filling'] = (df['close'] - df['open']) / abs(df['overnight_gap'] * df['close'].shift(1))
    df['gap_filling'] = np.where(df['overnight_gap'] > 0, -df['gap_filling'], df['gap_filling'])
    df['historical_gap_fill'] = df['gap_filling'].rolling(window=63).mean()
    df['mean_reversion'] = df['overnight_gap'] * df['historical_gap_fill']
    
    # Momentum Acceleration Deceleration
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_accel'] = df['momentum_5'] - df['momentum_10']
    df['volume_accel'] = df['volume'] / df['volume'].rolling(window=10).mean()
    df['acceleration_signal'] = df['momentum_accel'] * df['volume_accel']
    
    # Combine all signals with equal weights
    signals = [
        'regime_adj_momentum', 'breakout_strength', 'persistence_signal',
        'liquidity_adj_reversal', 'volume_divergence', 'pattern_signal',
        'mean_reversion', 'acceleration_signal'
    ]
    
    # Z-score normalization for each signal
    for signal in signals:
        df[f'{signal}_z'] = (df[signal] - df[signal].rolling(window=252).mean()) / df[signal].rolling(window=252).std()
    
    # Final combined factor
    df['alpha_factor'] = sum(df[f'{signal}_z'] for signal in signals) / len(signals)
    
    return df['alpha_factor']
