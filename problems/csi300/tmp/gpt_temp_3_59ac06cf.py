import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price Reversal with Volume Confirmation and Liquidity Dynamics
    """
    data = df.copy()
    
    # Multi-Timeframe Price Reversal Detection
    # Calculate rolling returns
    data['ret_3d'] = data['close'].pct_change(3)
    data['ret_10d'] = data['close'].pct_change(10)
    data['ret_30d'] = data['close'].pct_change(30)
    
    # Calculate return percentiles
    data['ret_3d_pct'] = data['ret_3d'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 5 else np.nan, raw=False
    )
    data['ret_10d_pct'] = data['ret_10d'].rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False
    )
    data['ret_30d_pct'] = data['ret_30d'].rolling(window=90, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 30 else np.nan, raw=False
    )
    
    # Identify reversal patterns
    data['overbought_3d'] = ((data['ret_3d_pct'] > 0.8) & (data['ret_3d'] > 0)).astype(int)
    data['oversold_3d'] = ((data['ret_3d_pct'] < 0.2) & (data['ret_3d'] < 0)).astype(int)
    data['overbought_10d'] = ((data['ret_10d_pct'] > 0.8) & (data['ret_10d'] > 0)).astype(int)
    data['oversold_10d'] = ((data['ret_10d_pct'] < 0.2) & (data['ret_10d'] < 0)).astype(int)
    data['overbought_30d'] = ((data['ret_30d_pct'] > 0.8) & (data['ret_30d'] > 0)).astype(int)
    data['oversold_30d'] = ((data['ret_30d_pct'] < 0.2) & (data['ret_30d'] < 0)).astype(int)
    
    # Calculate reversal strength
    data['rev_strength_3d'] = np.where(data['oversold_3d'] == 1, -data['ret_3d_pct'],
                                      np.where(data['overbought_3d'] == 1, -(1 - data['ret_3d_pct']), 0))
    data['rev_strength_10d'] = np.where(data['oversold_10d'] == 1, -data['ret_10d_pct'],
                                       np.where(data['overbought_10d'] == 1, -(1 - data['ret_10d_pct']), 0))
    data['rev_strength_30d'] = np.where(data['oversold_30d'] == 1, -data['ret_30d_pct'],
                                       np.where(data['overbought_30d'] == 1, -(1 - data['ret_30d_pct']), 0))
    
    # Assess consistency across timeframes
    data['consistency_score'] = (
        data[['overbought_3d', 'overbought_10d', 'overbought_30d']].sum(axis=1) +
        data[['oversold_3d', 'oversold_10d', 'oversold_30d']].sum(axis=1)
    ) / 3.0
    
    # Volume Confirmation Analysis
    # Volume surge detection
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_20d_median'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_surge_ratio'] = data['volume_3d_avg'] / data['volume_20d_median']
    data['volume_spike'] = (data['volume_surge_ratio'] > 2).astype(int)
    
    # Volume persistence
    data['volume_persistence'] = data['volume_spike'].rolling(window=5, min_periods=3).mean()
    
    # Price-Volume correlation
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['price_volume_corr'] = data['price_change'].rolling(window=10, min_periods=5).corr(data['volume_change'])
    
    # Price-Volume divergence
    data['pv_divergence'] = np.where(
        (data['price_change'] > 0) & (data['volume_change'] < 0), -1,
        np.where((data['price_change'] < 0) & (data['volume_change'] > 0), 1, 0)
    )
    
    # Liquidity Dynamics Assessment
    # Trade size analysis
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['trade_size_change'] = data['avg_trade_size'].pct_change()
    data['trade_size_trend'] = data['trade_size_change'].rolling(window=5, min_periods=3).mean()
    
    # Market depth proxy
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['price_impact'] = data['daily_range'] / (data['volume'] + 1e-8)
    data['volatility_volume_ratio'] = data['daily_range'].rolling(window=5, min_periods=3).std() / (data['volume'].rolling(window=5, min_periods=3).mean() + 1e-8)
    
    # Regime-Dependent Signal Processing
    # Liquidity regime classification
    data['liquidity_regime'] = np.where(
        data['volatility_volume_ratio'] < data['volatility_volume_ratio'].rolling(window=20, min_periods=10).quantile(0.3), 1,  # High liquidity
        np.where(data['volatility_volume_ratio'] > data['volatility_volume_ratio'].rolling(window=20, min_periods=10).quantile(0.7), -1, 0)  # Low liquidity
    )
    
    # Composite Alpha Construction
    # Base reversal component
    data['base_reversal'] = (
        data['rev_strength_3d'] * 0.4 + 
        data['rev_strength_10d'] * 0.35 + 
        data['rev_strength_30d'] * 0.25
    ) * (1 + data['consistency_score'])
    
    # Volume confirmation multiplier
    data['volume_multiplier'] = np.where(
        data['volume_spike'] == 1, 1.5,
        np.where(data['pv_divergence'] != 0, 0.7, 1.0)
    ) * (1 + 0.2 * data['volume_persistence'])
    
    # Liquidity adjustment
    data['liquidity_adjustment'] = np.where(
        data['liquidity_regime'] == 1, 1.3,  # High liquidity - enhance signals
        np.where(data['liquidity_regime'] == -1, 0.7, 1.0)  # Low liquidity - reduce signals
    ) * (1 + 0.1 * np.sign(data['trade_size_trend']))
    
    # Final composite factor
    data['alpha_factor'] = (
        data['base_reversal'] * 
        data['volume_multiplier'] * 
        data['liquidity_adjustment']
    )
    
    return data['alpha_factor']
