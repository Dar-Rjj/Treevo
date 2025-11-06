import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns and volatility
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    
    # Volatility regime classification using 20-day rolling percentiles
    df['vol_regime'] = pd.cut(df['volatility'].rolling(window=20).rank(pct=True), 
                             bins=[0, 0.33, 0.66, 1], labels=['low', 'medium', 'high'])
    
    # Volatility-adjusted momentum score
    # 5-day return normalized by current volatility regime's typical volatility
    df['mom_5d'] = df['close'].pct_change(5)
    
    # Calculate regime-specific volatility averages (using expanding window to avoid lookahead)
    regime_vol = {}
    for regime in ['low', 'medium', 'high']:
        mask = df['vol_regime'] == regime
        regime_vol[regime] = df.loc[mask, 'volatility'].expanding().mean()
    
    # Combine regime volatilities
    df['regime_vol'] = np.nan
    for regime, vol_series in regime_vol.items():
        df.loc[vol_series.index, 'regime_vol'] = vol_series
    
    # Volatility-adjusted momentum
    df['vol_adj_momentum'] = df['mom_5d'] / (df['regime_vol'] + 1e-8)
    
    # Momentum consistency across volatility conditions
    # Calculate momentum in different volatility states using expanding window
    df['mom_low_vol'] = df['mom_5d'].where(df['vol_regime'] == 'low').expanding().mean()
    df['mom_high_vol'] = df['mom_5d'].where(df['vol_regime'] == 'high').expanding().mean()
    df['momentum_consistency'] = 1 - abs(df['mom_low_vol'] - df['mom_high_vol']) / (abs(df['mom_low_vol']) + abs(df['mom_high_vol']) + 1e-8)
    
    # Price memory persistence
    # Calculate how often price behavior repeats in similar volatility regimes
    df['vol_regime_change'] = df['vol_regime'].ne(df['vol_regime'].shift(1))
    
    # Price behavior in current regime (expanding correlation between returns and past returns in same regime)
    same_regime_mask = df['vol_regime'] == df['vol_regime'].shift(1)
    df['regime_persistence'] = df['returns'].rolling(window=10).corr(df['returns'].shift(1).where(same_regime_mask))
    
    # Volume entropy during volatility shifts
    df['volume_change'] = df['volume'].pct_change()
    df['vol_shift_volume'] = df['volume_change'].where(df['vol_regime_change'])
    
    # Calculate entropy of volume changes during regime transitions (using rolling window)
    def rolling_entropy(series, window=10):
        def calc_entropy(x):
            x = x.dropna()
            if len(x) < 2:
                return np.nan
            hist, _ = np.histogram(x, bins=5, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        
        return series.rolling(window=window).apply(calc_entropy, raw=False)
    
    df['volume_entropy'] = rolling_entropy(df['vol_shift_volume'].fillna(0))
    
    # Combine components with volume weighting
    momentum_component = df['vol_adj_momentum'] * df['momentum_consistency']
    memory_component = df['regime_persistence'].fillna(0)
    
    # Volume confirmation weight (higher entropy = less reliable transitions)
    volume_weight = 1 / (1 + df['volume_entropy'].fillna(1))
    
    # Final factor combining momentum, memory, and volume confirmation
    factor = momentum_component * memory_component * volume_weight
    
    return factor
