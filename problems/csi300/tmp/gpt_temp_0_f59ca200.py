import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Factor that combines multi-timeframe volatility analysis,
    price-volume divergence dynamics, spread efficiency, and overnight gap momentum.
    """
    # Multi-Timeframe Volatility Regime Detection
    # Micro-regime: Intraday volatility clustering (t-1 to t)
    df['micro_vol'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['micro_vol_cluster'] = df['micro_vol'].rolling(window=2).std()
    
    # Meso-regime: Rolling volatility persistence (t-5 to t)
    df['meso_vol'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['close'].rolling(window=5).mean()
    df['vol_persistence'] = df['meso_vol'].rolling(window=3).apply(lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else 1)
    
    # Macro-regime: Structural volatility shifts (t-20 to t)
    df['macro_vol'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    df['vol_regime_shift'] = df['macro_vol'] / df['macro_vol'].rolling(window=10).mean()
    
    # Price-Volume Divergence Dynamics
    # Volume acceleration without price movement
    df['volume_accel'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['vol_price_divergence'] = df['volume_accel'] / (df['price_range'] + 1e-6)
    
    # Price momentum with declining volume participation
    df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['volume_trend'] = df['volume'].rolling(window=5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.mean() if x.mean() > 0 else 0)
    df['momentum_volume_div'] = df['price_momentum'] * (1 - np.tanh(df['volume_trend']))
    
    # Extreme volume spikes with minimal price impact
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / (df['volume'].rolling(window=20).std() + 1e-6)
    df['price_impact'] = abs(df['close'] - df['open']) / df['open']
    df['extreme_vol_inefficiency'] = np.where(
        abs(df['volume_zscore']) > 2,
        df['volume_zscore'] / (df['price_impact'] + 1e-6),
        0
    )
    
    # Bid-Ask Spread Efficiency Analysis (approximated using OHLC)
    # Spread width relative to price range
    df['effective_spread'] = 2 * abs((df['high'] + df['low']) / 2 - df['close'])
    df['spread_efficiency'] = df['effective_spread'] / (df['high'] - df['low'] + 1e-6)
    
    # Spread persistence during high volatility periods
    df['high_vol_period'] = df['micro_vol'] > df['micro_vol'].rolling(window=10).quantile(0.7)
    df['spread_persistence'] = df['spread_efficiency'].rolling(window=3).std()
    df['vol_spread_interaction'] = np.where(
        df['high_vol_period'],
        df['spread_persistence'] * df['micro_vol'],
        0
    )
    
    # Spread compression timing vs price movements
    df['spread_compression'] = df['spread_efficiency'] < df['spread_efficiency'].rolling(window=10).quantile(0.3)
    df['compression_momentum'] = np.where(
        df['spread_compression'],
        df['close'] / df['close'].shift(1) - 1,
        0
    )
    
    # Overnight Gap Momentum Persistence
    # Gap fill vs continuation probability assessment
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_fill_ratio'] = abs((df['close'] - df['open']) / (df['overnight_gap'] * df['close'].shift(1) + 1e-6))
    df['gap_persistence'] = np.where(
        df['overnight_gap'] > 0,
        np.minimum(1, df['gap_fill_ratio']),
        np.maximum(-1, -df['gap_fill_ratio'])
    )
    
    # Pre-gap volume accumulation patterns
    df['pre_gap_volume'] = df['volume'].shift(1) / df['volume'].rolling(window=5).mean().shift(1)
    df['volume_accumulation'] = df['pre_gap_volume'] * np.sign(df['overnight_gap'])
    
    # Post-gap volatility expansion characteristics
    df['post_gap_vol_expansion'] = df['micro_vol'] / df['micro_vol'].shift(1)
    df['gap_vol_dynamics'] = abs(df['overnight_gap']) * df['post_gap_vol_expansion']
    
    # Composite factor synthesis with volatility regime weighting
    volatility_weight = 1 / (1 + df['macro_vol'])
    
    # Core components
    divergence_component = (
        df['vol_price_divergence'].fillna(0) * 0.3 +
        df['momentum_volume_div'].fillna(0) * 0.3 +
        df['extreme_vol_inefficiency'].fillna(0) * 0.4
    )
    
    spread_component = (
        df['spread_efficiency'].fillna(0) * -0.4 +
        df['vol_spread_interaction'].fillna(0) * 0.3 +
        df['compression_momentum'].fillna(0) * 0.3
    )
    
    gap_component = (
        df['gap_persistence'].fillna(0) * 0.4 +
        df['volume_accumulation'].fillna(0) * 0.3 +
        df['gap_vol_dynamics'].fillna(0) * 0.3
    )
    
    # Final factor with regime adaptation
    factor = (
        divergence_component * 0.4 +
        spread_component * 0.3 +
        gap_component * 0.3
    ) * volatility_weight
    
    # Clean up and return
    result = pd.Series(factor, index=df.index)
    return result
