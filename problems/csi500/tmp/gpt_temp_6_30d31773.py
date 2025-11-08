import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Detection
    df['returns'] = df['close'] / df['close'].shift(1) - 1
    df['realized_vol_10d'] = df['returns'].rolling(window=10).std()
    df['vol_regime'] = df['realized_vol_10d'] > df['realized_vol_10d'].rolling(window=20).median()
    
    # Price Rejection Component with Volatility Sensitivity
    df['price_rejection'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['intraday_vol_change'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1)) - 1
    df['gap_direction'] = np.sign(df['open'] / df['close'].shift(1) - 1)
    df['volatility_weighted_rejection'] = df['price_rejection'] * df['intraday_vol_change'] * df['gap_direction']
    
    # Volume-Momentum Confirmation
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['oversold_condition'] = (df['momentum_5d'] < -0.03) & (df['momentum_10d'] < -0.02)
    
    df['market_depth'] = df['volume'] / ((df['high'] - df['low']) / df['close'])
    df['volume_sma_5d'] = df['volume'].rolling(window=5).mean()
    df['volume_surge'] = df['volume'] > (1.2 * df['volume_sma_5d'])
    
    df['signed_volume'] = df['volume'] * np.sign(df['close'] - df['open'])
    df['volume_price_convergence'] = (df['signed_volume'] * (df['close'] - df['open'])).rolling(window=3).sum()
    
    # Combine Momentum and Volume Components
    df['volume_momentum_component'] = df['oversold_condition'].astype(float)
    df['volume_momentum_component'] *= df['market_depth']
    df['volume_momentum_component'] *= np.where(df['volume_surge'], 1.5, 1.0)
    
    # Regime-Adaptive Factor Combination
    df['price_efficiency'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])).rolling(window=5).mean()
    
    # High Volatility Regime
    high_vol_component = df['volatility_weighted_rejection'] * df['volume_momentum_component'] * df['market_depth']
    
    # Low Volatility Regime
    low_vol_component = df['price_efficiency'] * df['volume_momentum_component'] * df['market_depth']
    
    df['regime_adaptive_factor'] = np.where(df['vol_regime'], high_vol_component, low_vol_component)
    
    # Historical Pattern Integration
    df['daily_returns'] = df['close'] / df['close'].shift(1) - 1
    df['reversal_pattern'] = df['daily_returns'].rolling(window=3).apply(
        lambda x: np.corrcoef(x.iloc[1:], x.iloc[:-1])[0,1] if len(x) == 3 and not np.isnan(x).any() else np.nan, 
        raw=False
    )
    
    # Volatility Ratio Adjustment
    df['vol_ratio'] = df['returns'].rolling(window=10).std() / df['returns'].rolling(window=20).std()
    
    # Final Alpha Generation
    df['alpha_raw'] = df['regime_adaptive_factor'] * abs(df['reversal_pattern']) * df['vol_ratio']
    
    return df['alpha_raw']
