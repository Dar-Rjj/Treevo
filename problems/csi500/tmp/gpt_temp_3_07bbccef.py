import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum Factor with multi-timeframe convergence
    """
    df = data.copy()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Volatility Regime Detection
    df['vol_5d'] = df['returns'].rolling(window=5).std()
    df['vol_20d'] = df['returns'].rolling(window=20).std()
    
    # True Range calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_3d'] = df['true_range'].rolling(window=3).mean()
    df['atr_10d'] = df['true_range'].rolling(window=10).mean()
    
    # Volatility regime classification
    df['vol_ratio'] = df['vol_5d'] / df['vol_20d']
    df['vol_change'] = df['vol_5d'].pct_change(3)
    
    conditions = [
        (df['vol_ratio'] > 1.2) & (abs(df['vol_change']) > 0.3),  # Transition
        df['vol_ratio'] > 1.0,  # High volatility
        df['vol_ratio'] <= 1.0   # Low volatility
    ]
    choices = [2, 1, 0]  # 2=Transition, 1=High, 0=Low
    df['vol_regime'] = np.select(conditions, choices, default=0)
    
    # Multi-timeframe momentum calculations
    # Short-term (1-3 days)
    df['mom_1d'] = df['close'].pct_change(1)
    df['mom_3d'] = df['close'].pct_change(3)
    df['volume_trend_3d'] = df['volume'].rolling(window=3).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0
    )
    
    # Medium-term (5-10 days)
    df['mom_5d'] = df['close'].pct_change(5)
    df['mom_10d'] = df['close'].pct_change(10)
    df['trend_strength_10d'] = df['close'].rolling(window=10).apply(
        lambda x: (x[-1] - x[0]) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )
    
    # Long-term (20+ days)
    df['mom_20d'] = df['close'].pct_change(20)
    df['position_in_range'] = (df['close'] - df['low'].rolling(window=20).min()) / \
                             (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min())
    
    # Volume-based features
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['avg_trade_size'] = df['amount'] / df['volume']
    df['volume_persistence'] = df['volume'].rolling(window=5).apply(
        lambda x: (x > x.mean()).sum() / len(x)
    )
    
    # Regime-specific momentum adjustments
    df['regime_adjusted_momentum'] = 0.0
    
    # High volatility regime
    high_vol_mask = df['vol_regime'] == 1
    df.loc[high_vol_mask, 'regime_adjusted_momentum'] = (
        df.loc[high_vol_mask, 'mom_1d'] / df.loc[high_vol_mask, 'vol_5d'] * 
        df.loc[high_vol_mask, 'volume_ratio']
    )
    
    # Low volatility regime
    low_vol_mask = df['vol_regime'] == 0
    df.loc[low_vol_mask, 'regime_adjusted_momentum'] = (
        df.loc[low_vol_mask, 'mom_5d'] * df.loc[low_vol_mask, 'vol_ratio'] * 
        df.loc[low_vol_mask, 'volume_persistence']
    )
    
    # Transition regime
    trans_mask = df['vol_regime'] == 2
    df.loc[trans_mask, 'regime_adjusted_momentum'] = (
        (df.loc[trans_mask, 'mom_1d'] + df.loc[trans_mask, 'mom_3d'] + 
         df.loc[trans_mask, 'mom_5d']) / 3 * 
        (1 + df.loc[trans_mask, 'volume_trend_3d'])
    )
    
    # Multi-timeframe convergence factor
    # Short-term weight (higher in high volatility)
    df['weight_short'] = np.where(df['vol_regime'] == 1, 0.5, 
                                 np.where(df['vol_regime'] == 2, 0.4, 0.2))
    
    # Medium-term weight (balanced across regimes)
    df['weight_medium'] = 0.4
    
    # Long-term weight (higher in low volatility)
    df['weight_long'] = np.where(df['vol_regime'] == 0, 0.4, 
                                np.where(df['vol_regime'] == 2, 0.2, 0.1))
    
    # Normalize momentum signals
    df['mom_short_norm'] = df['mom_3d'] / df['vol_5d'].replace(0, 1e-6)
    df['mom_medium_norm'] = df['mom_10d'] / df['vol_20d'].replace(0, 1e-6)
    df['mom_long_norm'] = df['mom_20d'] * (2 * df['position_in_range'] - 1)
    
    # Final convergence factor
    df['convergence_factor'] = (
        df['weight_short'] * df['mom_short_norm'] +
        df['weight_medium'] * df['mom_medium_norm'] +
        df['weight_long'] * df['mom_long_norm']
    )
    
    # Combine regime-adjusted momentum with convergence factor
    df['final_alpha'] = (
        0.6 * df['regime_adjusted_momentum'] + 
        0.4 * df['convergence_factor']
    )
    
    # Liquidity adjustment
    df['liquidity_score'] = (
        (1 / (df['high'] - df['low']).replace(0, 1e-6)) *  # Inverse of spread
        df['avg_trade_size'] *  # Market depth proxy
        (abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-6))  # Range efficiency
    )
    
    # Final alpha with liquidity adjustment
    df['alpha'] = df['final_alpha'] * df['liquidity_score'].rolling(window=5).mean()
    
    return df['alpha']
