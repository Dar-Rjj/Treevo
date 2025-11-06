import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Enhanced momentum-volume synergy with microstructure adaptation
    # Focus on convergence of price and volume signals with volatility sensitivity
    
    # Multi-timeframe momentum convergence
    mom_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_8 = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    mom_15 = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    momentum_triple = (mom_3 + mom_8 + mom_15) / 3
    momentum_divergence = mom_3 - mom_15
    
    # Volume momentum with acceleration
    vol_ma_5 = df['volume'].rolling(window=5).mean()
    vol_ma_15 = df['volume'].rolling(window=15).mean()
    volume_accel = (df['volume'] / vol_ma_5) - (df['volume'] / vol_ma_15)
    volume_persistence = df['volume'].rolling(window=5).apply(lambda x: (x > x.shift(1)).sum())
    
    # Microstructure efficiency signals
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    gap_efficiency = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    price_consistency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Adaptive volatility normalization
    atr_5 = (df['high'] - df['low']).rolling(window=5).mean()
    atr_10 = (df['high'] - df['low']).rolling(window=10).mean()
    volatility_regime = atr_5 / atr_10
    volatility_sensitivity = 1 / (atr_5 + atr_10 + 1e-7)
    
    # Volume-price synergy factors
    price_vol_corr = df['close'].pct_change().rolling(window=8).corr(df['volume'].pct_change())
    amount_intensity = df['amount'] / (df['volume'] * df['close'] + 1e-7)
    
    # Composite factor construction
    momentum_volume_synergy = momentum_triple * (1 + volume_accel) * (1 + volume_persistence/5)
    microstructure_quality = intraday_efficiency * gap_efficiency * price_consistency
    volatility_adaptation = volatility_sensitivity * (1 + volatility_regime)
    
    factor = (momentum_volume_synergy * microstructure_quality * volatility_adaptation * 
             (1 + price_vol_corr) * (1 + amount_intensity))
    
    return factor
