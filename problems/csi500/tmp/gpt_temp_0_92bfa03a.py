import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate composite alpha factor combining multiple market microstructure signals
    """
    df = data.copy()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # 1. Intraday Momentum Persistence Factor
    df['prev_close'] = df['close'].shift(1)
    df['range_momentum'] = (df['high'] - df['low']) / df['prev_close']
    df['price_momentum'] = (df['close'] - df['prev_close']) / df['prev_close']
    
    # Combined momentum
    df['combined_momentum'] = df['range_momentum'] * df['price_momentum']
    
    # Track consecutive same-direction days
    df['momentum_direction'] = np.sign(df['price_momentum'])
    df['consecutive_days'] = 0
    for i in range(1, len(df)):
        if df['momentum_direction'].iloc[i] == df['momentum_direction'].iloc[i-1]:
            df.loc[df.index[i], 'consecutive_days'] = df['consecutive_days'].iloc[i-1] + 1
    
    # Volume confirmation
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['returns_5d_std'] = df['price_momentum'].rolling(window=5).std()
    
    # Composite signal 1
    df['signal_1'] = (df['combined_momentum'] * df['volume_ratio'] * 
                     (df['consecutive_days'] + 1)) / df['returns_5d_std'].replace(0, np.nan)
    
    # 2. Volatility Regime Adaptive Factor
    df['vol_short'] = df['price_momentum'].rolling(window=5).std()
    df['vol_long'] = df['price_momentum'].rolling(window=20).std()
    df['vol_ratio'] = df['vol_short'] / df['vol_long']
    
    # Multi-timeframe momentum
    df['ret_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['ret_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['ret_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Regime-based weighting
    conditions = [
        df['vol_ratio'] > 1.2,
        df['vol_ratio'] < 0.8,
        True  # normal volatility
    ]
    choices = [
        df['ret_3d'] * 0.6 + df['ret_10d'] * 0.3 + df['ret_20d'] * 0.1,  # high vol: emphasize short-term
        df['ret_3d'] * 0.1 + df['ret_10d'] * 0.3 + df['ret_20d'] * 0.6,  # low vol: emphasize long-term
        df['ret_3d'] * 0.33 + df['ret_10d'] * 0.33 + df['ret_20d'] * 0.34  # normal: equal weighting
    ]
    df['regime_momentum'] = np.select(conditions, choices, default=0)
    
    # Volume validation
    df['vol_20d_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_condition'] = np.where(df['volume'] > df['vol_20d_avg'], 1.2, 0.8)
    df['signal_2'] = df['regime_momentum'] * df['volume_condition']
    
    # 3. Liquidity-Efficient Reversal Factor
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['price_deviation'] = (df['close'] - df['mid_price']) / ((df['high'] - df['low']) / 2)
    df['liquidity_ratio'] = df['amount'] / (df['high'] - df['low']).replace(0, np.nan)
    df['liquidity_10d_avg'] = df['liquidity_ratio'].rolling(window=10).mean()
    df['liquidity_comparison'] = df['liquidity_ratio'] / df['liquidity_10d_avg']
    
    # Reversal signal with volume confirmation
    df['reversal_signal'] = -df['price_deviation'] * df['liquidity_comparison']
    df['vol_5d_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_surge'] = np.where(df['volume'] > df['vol_5d_avg'], 1.5, 1.0)
    df['signal_3'] = df['reversal_signal'] * df['volume_surge']
    
    # 4. Order Flow Momentum Factor
    df['amount_direction'] = np.where(df['close'] > df['open'], df['amount'], -df['amount'])
    df['net_amount_cum'] = df['amount_direction'].rolling(window=5).sum()
    
    # Flow-return correlation
    df['flow_correlation'] = df['amount_direction'].rolling(window=5).corr(df['price_momentum'])
    
    # Generate signal
    df['signal_4'] = df['net_amount_cum'] * df['flow_correlation'].fillna(0)
    
    # 5. Breakout Confidence Factor
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['prev_close'] - df['low'])
        )
    )
    df['atr_10d'] = df['true_range'].rolling(window=10).mean()
    df['breakout_ratio'] = df['true_range'] / df['atr_10d']
    
    # Volume validation
    df['volume_surge_20d'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Volume-range correlation
    df['vol_range_corr'] = df['volume'].rolling(window=5).corr(df['true_range'])
    
    # Confidence score
    df['signal_5'] = (df['breakout_ratio'] * df['volume_surge_20d'] * 
                     df['vol_range_corr'].abs().fillna(0))
    
    # Combine all signals with equal weighting
    signals = ['signal_1', 'signal_2', 'signal_3', 'signal_4', 'signal_5']
    for signal in signals:
        df[signal] = df[signal].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Z-score normalization for each signal
    for signal in signals:
        mean_val = df[signal].mean()
        std_val = df[signal].std()
        if std_val > 0:
            df[f'{signal}_norm'] = (df[signal] - mean_val) / std_val
        else:
            df[f'{signal}_norm'] = 0
    
    # Final composite factor
    result = (df['signal_1_norm'] + df['signal_2_norm'] + df['signal_3_norm'] + 
              df['signal_4_norm'] + df['signal_5_norm']) / 5
    
    return result
