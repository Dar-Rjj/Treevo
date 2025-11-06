import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Scale Divergence Momentum with Efficiency Confirmation
    Combines volatility-liquidity divergence analysis with price efficiency momentum
    and quality assessment to generate an interpretable alpha factor.
    """
    df = data.copy()
    
    # Calculate True Range
    df['TR'] = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1))
        ),
        abs(df['low'] - df['close'].shift(1))
    )
    
    # Volatility-Liquidity Divergence Analysis
    # Short-term divergence (3-day)
    df['vol_momentum_3d'] = df['TR'].rolling(window=3).mean() / df['TR'].shift(3).rolling(window=3).mean() - 1
    df['liq_momentum_3d'] = df['amount'].rolling(window=3).mean() / df['amount'].shift(3).rolling(window=3).mean() - 1
    df['divergence_3d'] = np.sign(df['vol_momentum_3d']) * np.sign(df['liq_momentum_3d'])
    
    # Medium-term divergence (5-day)
    df['vol_momentum_5d'] = df['TR'].rolling(window=5).mean() / df['TR'].shift(5).rolling(window=5).mean() - 1
    df['liq_momentum_5d'] = df['amount'].rolling(window=5).mean() / df['amount'].shift(5).rolling(window=5).mean() - 1
    df['divergence_5d'] = (abs(df['vol_momentum_5d']) - abs(df['liq_momentum_5d'])) / (abs(df['vol_momentum_5d']) + abs(df['liq_momentum_5d']) + 1e-8)
    
    # Long-term divergence (10-day)
    df['vol_momentum_10d'] = df['TR'].rolling(window=10).mean() / df['TR'].shift(10).rolling(window=10).mean() - 1
    df['liq_momentum_10d'] = df['amount'].rolling(window=10).mean() / df['amount'].shift(10).rolling(window=10).mean() - 1
    df['divergence_10d'] = np.sign(df['vol_momentum_10d']) * np.sign(df['liq_momentum_10d'])
    
    # Trend consistency across periods
    df['trend_consistency'] = (df['divergence_3d'] + df['divergence_5d'] + df['divergence_10d']) / 3
    
    # Price Efficiency Momentum
    # Range Efficiency
    df['daily_range_eff'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['range_eff_5d'] = df['daily_range_eff'].rolling(window=5).mean()
    df['eff_momentum'] = df['range_eff_5d'] / df['range_eff_5d'].shift(5) - 1
    
    # Movement Efficiency
    df['net_movement'] = abs(df['close'] - df['close'].shift(5))
    df['total_movement'] = abs(df['close'] - df['close'].shift(1)).rolling(window=5).sum()
    df['price_efficiency'] = df['net_movement'] / (df['total_movement'] + 1e-8)
    
    # Intraday Pattern Persistence
    df['intraday_strength'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['pattern_persistence'] = df['intraday_strength'].rolling(window=3).corr(df['intraday_strength'].shift(1))
    df['volume_autocorr'] = df['volume'].rolling(window=5).corr(df['volume'].shift(1))
    
    # Momentum Quality Assessment
    # Directional Consistency
    df['daily_return'] = df['close'].pct_change()
    df['return_5d'] = df['close'] / df['close'].shift(5) - 1
    df['sign_5d'] = np.sign(df['return_5d'])
    df['dir_consistency'] = (df['daily_return'].rolling(window=5).apply(
        lambda x: np.sum(np.sign(x) == df.loc[x.index[-1], 'sign_5d']) if not pd.isna(df.loc[x.index[-1], 'sign_5d']) else 0
    )) / 5
    
    # Momentum Stability
    df['return_variance'] = df['daily_return'].rolling(window=5).var()
    df['momentum_stability'] = abs(df['return_5d']) / (df['return_variance'] + 0.0001)
    
    # Volume Confirmation
    df['volume_trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=10).mean()
    df['volume_breakout'] = df['volume'] / df['volume'].rolling(window=50).mean()
    df['volume_rank'] = df['volume'].rolling(window=20).rank(pct=True)
    
    # Volatility Regime Context
    df['vol_ratio'] = df['TR'].rolling(window=5).mean() / df['TR'].rolling(window=20).mean()
    
    # Regime-based momentum selection
    conditions = [
        df['vol_ratio'] > 1.2,
        (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 1.2),
        df['vol_ratio'] < 0.8
    ]
    choices = [
        df['close'] / df['close'].shift(5) - 1,  # 5-day momentum
        df['close'] / df['close'].shift(10) - 1,  # 10-day momentum
        df['close'] / df['close'].shift(20) - 1   # 20-day momentum
    ]
    df['regime_momentum'] = np.select(conditions, choices, default=0)
    
    # Alpha Factor Synthesis
    # Multi-Scale Divergence Score
    divergence_weights = np.array([0.3, 0.4, 0.3])  # weights for 3d, 5d, 10d
    df['divergence_score'] = (
        divergence_weights[0] * df['divergence_3d'] * abs(df['liq_momentum_3d']) +
        divergence_weights[1] * df['divergence_5d'] * abs(df['liq_momentum_5d']) +
        divergence_weights[2] * df['divergence_10d'] * abs(df['liq_momentum_10d'])
    ) * (1 + df['trend_consistency']) * (1 + df['eff_momentum'])
    
    # Quality-Enhanced Momentum
    df['quality_momentum'] = (
        df['regime_momentum'] * 
        df['dir_consistency'] * 
        df['momentum_stability'] * 
        df['volume_trend'] * 
        df['volume_breakout']
    )
    
    # Final Factor Generation
    df['alpha_factor'] = (
        df['divergence_score'] * 
        df['quality_momentum'] * 
        (1 + df['pattern_persistence']) * 
        df['price_efficiency'] * 
        df['volume_autocorr']
    )
    
    return df['alpha_factor']
