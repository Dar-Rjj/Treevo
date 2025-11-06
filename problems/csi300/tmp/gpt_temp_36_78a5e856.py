import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATRs for volatility regime
    df['atr_5'] = df['true_range'].rolling(window=5).mean()
    df['atr_20'] = df['true_range'].rolling(window=20).mean()
    df['volatility_ratio'] = df['atr_5'] / df['atr_20']
    
    # Calculate momentum components
    df['momentum_5'] = (df['close'] / df['close'].shift(5)) - 1
    df['momentum_10'] = (df['close'] / df['close'].shift(10)) - 1
    
    # Volatility regime adjusted momentum
    df['regime_momentum'] = np.where(
        df['volatility_ratio'] > 1,
        df['momentum_5'] * (1 + df['momentum_5'] - df['momentum_5'].shift(1)),  # Acceleration in high vol
        df['momentum_10']  # Medium-term in low vol
    )
    
    # Calculate VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    
    # VWAP trends
    df['vwap_trend_5'] = (df['vwap'] / df['vwap'].shift(5)) - 1
    df['vwap_trend_10'] = (df['vwap'] / df['vwap'].shift(10)) - 1
    
    # Price efficiency
    df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume trend (5-day slope)
    df['volume_trend'] = df['volume'].rolling(window=5).apply(
        lambda x: (x[-1] - x[0]) / (len(x) - 1) if len(x) == 5 else np.nan
    )
    
    # Efficiency-weighted alignment
    df['efficiency_volume_alignment'] = df['price_efficiency'] * df['volume_trend']
    
    # Convergence strength
    df['vwap_convergence'] = (df['vwap_trend_5'] + df['vwap_trend_10']) / 2
    df['momentum_vwap_alignment'] = df['regime_momentum'] * df['vwap_convergence']
    
    # Volatility persistence (True Range auto-correlation)
    df['tr_autocorr'] = df['true_range'].rolling(window=20).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 20 else np.nan
    )
    
    # Recent price-volume divergence
    df['price_change_5'] = (df['close'] / df['close'].shift(5)) - 1
    df['volume_change_5'] = (df['volume'] / df['volume'].shift(5)) - 1
    df['price_volume_divergence'] = df['price_change_5'] - df['volume_change_5']
    
    # Amount-weighted confirmation
    df['amount_weighted_confirmation'] = df['price_volume_divergence'] * df['amount']
    
    # Validate momentum persistence
    df['momentum_persistence'] = df['regime_momentum'].rolling(window=5).apply(
        lambda x: 1 if all(np.diff(x) > 0) else (-1 if all(np.diff(x) < 0) else 0) if len(x) == 5 else 0
    )
    
    # Synthesize final factor
    df['efficiency_confidence'] = df['efficiency_volume_alignment'].rolling(window=5).mean()
    
    # Final alpha factor
    df['alpha_factor'] = (
        df['regime_momentum'] * 
        (1 + df['efficiency_volume_alignment']) * 
        (1 + df['amount_weighted_confirmation'].rolling(window=5).mean()) *
        df['volatility_ratio'] *
        (1 + df['efficiency_confidence']) *
        (1 + df['momentum_persistence'] * 0.1)
    )
    
    # Clean up intermediate columns
    result = df['alpha_factor'].copy()
    
    return result
