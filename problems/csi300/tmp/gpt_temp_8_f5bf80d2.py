import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum & Reversion
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['vol_adj_momentum'] = df['momentum_5d'] / (df['daily_range'].rolling(5).std() + 1e-6)
    df['reversion_signal'] = -df['momentum_10d'] / (df['daily_range'].rolling(10).std() + 1e-6)
    
    # Volume-Price Alignment
    df['volume_roc_5d'] = df['volume'].pct_change(5)
    df['price_volume_divergence'] = np.sign(df['momentum_5d']) * np.sign(df['volume_roc_5d'])
    df['volume_confirmation'] = df['price_volume_divergence'] * np.abs(df['volume_roc_5d'])
    df['volume_weighted_signal'] = df['vol_adj_momentum'] * df['volume_confirmation']
    
    # Intraday Efficiency Patterns
    df['price_efficiency'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
    
    # Efficiency persistence tracking with decay
    efficiency_threshold = df['price_efficiency'].rolling(20).median()
    efficient_days = (df['price_efficiency'] > efficiency_threshold).astype(int)
    inefficient_days = (df['price_efficiency'] < efficiency_threshold).astype(int)
    
    decay_weights = np.exp(-np.arange(10) / 3)  # Exponential decay
    df['efficient_persistence'] = efficient_days.rolling(10).apply(
        lambda x: np.sum(x * decay_weights[:len(x)]), raw=False
    )
    df['inefficient_persistence'] = inefficient_days.rolling(10).apply(
        lambda x: np.sum(x * decay_weights[:len(x)]), raw=False
    )
    
    df['efficiency_regime_signal'] = np.where(
        df['efficient_persistence'] > 0.5,
        df['volume_weighted_signal'],  # Trend continuation
        -df['reversion_signal']  # Reversal signal
    )
    
    # Range Breakout & Regime Adaptation
    df['true_range'] = np.maximum(
        np.maximum(df['high'] - df['low'], 
                  np.abs(df['high'] - df['close'].shift(1))),
        np.abs(df['low'] - df['close'].shift(1))
    )
    df['atr_10d'] = df['true_range'].rolling(10).mean()
    df['volatility_regime'] = df['atr_10d'] > df['atr_10d'].rolling(20).median()
    
    df['range_breakout'] = (df['high'] - df['low']) / df['atr_10d']
    df['breakout_signal'] = np.where(
        df['volatility_regime'],
        df['range_breakout'] * df['momentum_5d'],  # Strong breakout weighting
        df['range_breakout'] * 0.5 * df['momentum_5d']  # Conservative signals
    )
    
    # Multi-Timeframe Confirmation
    df['momentum_5d_rank'] = df['momentum_5d'].rolling(20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-6), raw=False
    )
    df['momentum_10d_rank'] = df['momentum_10d'].rolling(20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-6), raw=False
    )
    
    timeframe_alignment = np.sign(df['momentum_5d_rank']) == np.sign(df['momentum_10d_rank'])
    
    # Signal Integration & Alpha Generation
    signals = pd.DataFrame({
        'momentum': df['vol_adj_momentum'],
        'reversion': df['reversion_signal'],
        'volume_aligned': df['volume_weighted_signal'],
        'efficiency': df['efficiency_regime_signal'],
        'breakout': df['breakout_signal']
    })
    
    # Equal weighting with timeframe confirmation filter
    composite_score = signals.mean(axis=1)
    alpha = composite_score * timeframe_alignment
    
    # Final normalization
    alpha = (alpha - alpha.rolling(20).mean()) / (alpha.rolling(20).std() + 1e-6)
    
    return alpha
