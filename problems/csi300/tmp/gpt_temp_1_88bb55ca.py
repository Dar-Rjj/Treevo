import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Volume Regime Transition Framework alpha factor
    Identifies regime transitions through multi-scale volatility analysis and volume flow patterns
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Multi-Scale Volatility Regime Identification
    # Micro-Volatility: Intraday volatility normalized by previous close
    data['micro_vol'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Meso-Volatility: 5-day rolling volatility normalized by mean price
    data['meso_vol'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    
    # Macro-Volatility: 20-day range volatility normalized by 20-day ago close
    data['macro_vol'] = (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()) / data['close'].shift(20)
    
    # Volatility Regime Transition Detection
    # Volatility compression: consecutive decreasing micro volatility
    data['vol_compression'] = data['micro_vol'].rolling(window=3).apply(
        lambda x: 1 if (x.iloc[0] > x.iloc[1] > x.iloc[2]) else 0, raw=False
    )
    
    # Volatility expansion: sudden regime shift (micro vol > 2x rolling median)
    vol_median = data['micro_vol'].rolling(window=10).median()
    data['vol_expansion'] = (data['micro_vol'] > 2 * vol_median).astype(int)
    
    # Volume Flow Pattern Analysis
    # Volume asymmetry: directional volume momentum
    data['volume_asymmetry'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'] + data['volume'].shift(1))
    
    # Volume persistence: consecutive directional volume changes
    volume_change = data['volume'].diff()
    data['volume_persistence'] = volume_change.rolling(window=3).apply(
        lambda x: 1 if (x.iloc[0] > 0 and x.iloc[1] > 0 and x.iloc[2] > 0) else 
                  (-1 if (x.iloc[0] < 0 and x.iloc[1] < 0 and x.iloc[2] < 0) else 0), raw=False
    )
    
    # Volume extremes: current volume relative to 20-day quantiles
    vol_q80 = data['volume'].rolling(window=20).quantile(0.8)
    vol_q20 = data['volume'].rolling(window=20).quantile(0.2)
    data['volume_extremes'] = np.where(
        data['volume'] > vol_q80, 1,
        np.where(data['volume'] < vol_q20, -1, 0)
    )
    
    # Volume-Volatility Efficiency: price movement per unit volume
    price_range = data['high'] - data['low']
    data['vol_vol_efficiency'] = price_range / (data['volume'] + 1e-8)
    
    # Flow consistency: directional persistence across 3 time intervals
    close_change = data['close'].pct_change()
    vol_change = data['volume'].pct_change()
    data['flow_consistency'] = (
        (close_change.rolling(window=3).apply(lambda x: len([i for i in x if i > 0]), raw=False) / 3) *
        (vol_change.rolling(window=3).apply(lambda x: len([i for i in x if i > 0]), raw=False) / 3)
    )
    
    # Regime Synchronization Framework
    # Volatility-Volume Phase Analysis
    vol_median_20 = data['micro_vol'].rolling(window=20).median()
    vol_median_5 = data['volume'].rolling(window=20).median()
    
    # High Volatility + High Volume: Breakout confirmation
    data['breakout_confirmation'] = (
        (data['micro_vol'] > vol_median_20) & 
        (data['volume'] > vol_median_5)
    ).astype(int)
    
    # Low Volatility + High Volume: Accumulation/distribution
    data['accumulation_signal'] = (
        (data['micro_vol'] < vol_median_20) & 
        (data['volume'] > vol_median_5)
    ).astype(int)
    
    # Volatility-Volume Divergence: Regime transition signals
    vol_zscore = (data['micro_vol'] - data['micro_vol'].rolling(window=20).mean()) / data['micro_vol'].rolling(window=20).std()
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['vol_volume_divergence'] = vol_zscore - volume_zscore
    
    # Multi-Timeframe Alignment
    # Cross-scale convergence: aligned volatility regimes
    micro_vol_regime = (data['micro_vol'] > data['micro_vol'].rolling(window=20).median()).astype(int)
    meso_vol_regime = (data['meso_vol'] > data['meso_vol'].rolling(window=20).median()).astype(int)
    macro_vol_regime = (data['macro_vol'] > data['macro_vol'].rolling(window=20).median()).astype(int)
    
    data['cross_scale_convergence'] = (micro_vol_regime + meso_vol_regime + macro_vol_regime) / 3
    
    # Scale divergence: conflicting regime signals
    data['scale_divergence'] = (
        (micro_vol_regime != meso_vol_regime).astype(int) +
        (micro_vol_regime != macro_vol_regime).astype(int) +
        (meso_vol_regime != macro_vol_regime).astype(int)
    ) / 3
    
    # Transition-Based Prediction Factor
    # Breakout Anticipation: Volatility Compression + Volume Accumulation
    data['breakout_anticipation'] = (
        data['vol_compression'] * 
        data['accumulation_signal'] * 
        (1 - data['scale_divergence'])
    )
    
    # Mean Reversion Signals: Volatility Expansion + Volume Exhaustion
    volume_exhaustion = (data['volume'] < data['volume'].rolling(window=10).quantile(0.3)).astype(int)
    data['mean_reversion_signal'] = (
        data['vol_expansion'] * 
        volume_exhaustion * 
        data['scale_divergence']
    )
    
    # Final alpha factor: Combined regime transition signal
    alpha = (
        0.3 * data['breakout_anticipation'] +
        0.2 * data['mean_reversion_signal'] +
        0.15 * data['vol_volume_divergence'] +
        0.15 * data['cross_scale_convergence'] +
        0.1 * data['flow_consistency'] +
        0.1 * data['volume_persistence']
    )
    
    # Clean up any infinite or NaN values
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = alpha.fillna(method='ffill').fillna(0)
    
    return alpha
