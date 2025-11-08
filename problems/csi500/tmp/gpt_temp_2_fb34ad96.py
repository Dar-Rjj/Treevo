import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price and volume metrics
    df = df.copy()
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['turnover'] = df['amount'] / (df['volume'] * df['close'])
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate sector-level liquidity proxies (using rolling median as sector proxy)
    sector_window = 20
    df['sector_price_range'] = df['price_range'].rolling(window=sector_window, min_periods=10).median()
    df['sector_turnover'] = df['turnover'].rolling(window=sector_window, min_periods=10).median()
    
    # Sector liquidity score (higher score = better liquidity)
    df['sector_liquidity_score'] = (1 / (df['sector_price_range'] + 1e-8)) * df['sector_turnover']
    
    # Individual stock liquidity deviation
    df['liquidity_range_dev'] = df['price_range'] / (df['sector_price_range'] + 1e-8)
    df['liquidity_turnover_dev'] = df['turnover'] / (df['sector_turnover'] + 1e-8)
    df['liquidity_premium'] = (1 / df['liquidity_range_dev']) * df['liquidity_turnover_dev']
    
    # Liquidity regime classification
    df['liquidity_regime'] = 0  # 0: transitional, 1: high, -1: low
    high_liquidity_threshold = df['sector_liquidity_score'].rolling(window=50, min_periods=20).quantile(0.7)
    low_liquidity_threshold = df['sector_liquidity_score'].rolling(window=50, min_periods=20).quantile(0.3)
    
    df.loc[df['sector_liquidity_score'] > high_liquidity_threshold, 'liquidity_regime'] = 1
    df.loc[df['sector_liquidity_score'] < low_liquidity_threshold, 'liquidity_regime'] = -1
    
    # Microstructure momentum signals
    # Intraday price path efficiency
    gap_up_mask = df['open'] > df['close'].shift(1)
    gap_down_mask = df['open'] < df['close'].shift(1)
    
    df['gap_absorption'] = 0
    df.loc[gap_up_mask, 'gap_absorption'] = (df['high'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df.loc[gap_down_mask, 'gap_absorption'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    df['midday_reversal'] = np.abs((df['high'] + df['low']) / 2 - df['close']) / (df['high'] - df['low'] + 1e-8)
    
    # Volume-time distribution (using first/last hour approximation)
    df['early_late_volume_ratio'] = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: x[:3].mean() / (x[-2:].mean() + 1e-8) if len(x) >= 5 else np.nan
    )
    
    # Order flow imbalance proxy
    df['price_amplified_volume'] = df['volume'] * np.abs(df['close'] - df['open']) / (df['close'] + 1e-8)
    df['directional_volume'] = np.sign(df['close'] - df['open']) * df['volume']
    
    # Cross-asset liquidity momentum
    df['sector_liquidity_momentum'] = df['sector_liquidity_score'].pct_change(periods=5)
    df['liquidity_convergence'] = (df['sector_price_range'] - df['price_range']).rolling(window=5, min_periods=3).mean()
    
    # Cross-sectional liquidity factor
    df['relative_liquidity_rank'] = df['liquidity_premium'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan
    )
    
    # Adaptive microstructure alpha - regime-dependent signals
    # High liquidity regime signals
    high_liquidity_signals = (
        df['early_late_volume_ratio'].rolling(window=5, min_periods=3).mean() +
        df['gap_absorption'].rolling(window=5, min_periods=3).mean() -
        df['midday_reversal'].rolling(window=5, min_periods=3).mean()
    )
    
    # Low liquidity regime signals  
    low_liquidity_signals = (
        df['liquidity_convergence'].rolling(window=5, min_periods=3).mean() +
        df['relative_liquidity_rank'].rolling(window=5, min_periods=3).mean() +
        (1 - df['midday_reversal']).rolling(window=5, min_periods=3).mean()
    )
    
    # Transitional regime signals
    transitional_signals = (
        df['sector_liquidity_momentum'].rolling(window=5, min_periods=3).mean() +
        df['directional_volume'].pct_change(periods=3).rolling(window=5, min_periods=3).mean() +
        df['price_amplified_volume'].pct_change(periods=3).rolling(window=5, min_periods=3).mean()
    )
    
    # Final alpha factor with regime-dependent weighting
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    # Apply regime-specific weights
    high_liquidity_weight = 0.6
    low_liquidity_weight = 0.7  
    transitional_weight = 0.5
    
    alpha_factor = np.where(
        df['liquidity_regime'] == 1,
        high_liquidity_signals * high_liquidity_weight,
        np.where(
            df['liquidity_regime'] == -1,
            low_liquidity_signals * low_liquidity_weight,
            transitional_signals * transitional_weight
        )
    )
    
    # Add cross-asset confirmation
    cross_asset_confirmation = (
        df['sector_liquidity_momentum'].rolling(window=3, min_periods=2).mean() *
        df['relative_liquidity_rank'].rolling(window=3, min_periods=2).mean()
    )
    
    final_alpha = alpha_factor * (1 + 0.3 * cross_asset_confirmation)
    
    return pd.Series(final_alpha, index=df.index)
