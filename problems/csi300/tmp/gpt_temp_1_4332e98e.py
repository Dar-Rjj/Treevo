import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum convergence with volume-amount-price regime alignment
    # and adaptive volatility scaling using regime-dependent signal weighting
    
    # Multi-timeframe momentum (1, 3, 6, 12 days)
    mom_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_6d = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)
    mom_12d = (df['close'] - df['close'].shift(12)) / df['close'].shift(12)
    
    # Price efficiency across timeframes
    range_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    range_efficiency_3d = range_efficiency.rolling(window=3).mean()
    range_efficiency_6d = range_efficiency.rolling(window=6).mean()
    
    # Volume-amount-price regime detection
    volume_ma_short = df['volume'].rolling(window=3).mean()
    volume_ma_long = df['volume'].rolling(window=12).mean()
    amount_ma_short = df['amount'].rolling(window=3).mean()
    amount_ma_long = df['amount'].rolling(window=12).mean()
    price_ma_short = df['close'].rolling(window=3).mean()
    price_ma_long = df['close'].rolling(window=12).mean()
    
    # Multi-dimensional regime classification
    volume_regime = (df['volume'] > volume_ma_long).astype(int)
    amount_regime = (df['amount'] > amount_ma_long).astype(int)
    price_regime = (df['close'] > price_ma_long).astype(int)
    
    # Regime alignment score (0-3: number of aligned regimes)
    regime_alignment = volume_regime + amount_regime + price_regime
    
    # Volume-amount efficiency (trading intensity per unit volume)
    volume_amount_efficiency = df['amount'] / (df['volume'] + 1e-7)
    vae_ma_short = volume_amount_efficiency.rolling(window=3).mean()
    vae_ma_long = volume_amount_efficiency.rolling(window=12).mean()
    
    # Adaptive volatility regime with multi-timeframe detection
    intraday_range = (df['high'] - df['low']) / df['close']
    intraday_vol_short = intraday_range.rolling(window=3).mean()
    intraday_vol_long = intraday_range.rolling(window=12).mean()
    
    close_returns = df['close'].pct_change()
    close_vol_short = close_returns.rolling(window=3).std()
    close_vol_long = close_returns.rolling(window=12).std()
    
    # Volatility regime classification
    vol_regime_short = (intraday_vol_short > intraday_vol_long).astype(int)
    vol_regime_long = (close_vol_short > close_vol_long).astype(int)
    
    # Regime-dependent momentum convergence
    momentum_convergence = pd.Series(index=df.index, dtype=float)
    regime_weighting = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 12:
            # Current momentum values
            moms = [mom_1d.iloc[i], mom_3d.iloc[i], mom_6d.iloc[i], mom_12d.iloc[i]]
            positive_moms = [m for m in moms if m > 0]
            negative_moms = [m for m in moms if m < 0]
            
            # Regime-based signal construction
            current_regime = regime_alignment.iloc[i]
            vol_regime = vol_regime_short.iloc[i] + vol_regime_long.iloc[i]
            
            if current_regime == 3:  # All regimes aligned (strong signal environment)
                if len(positive_moms) >= 3:  # Strong bullish convergence
                    momentum_convergence.iloc[i] = (abs(mom_1d.iloc[i]) * abs(mom_3d.iloc[i]) * 
                                                  abs(mom_6d.iloc[i]) * abs(mom_12d.iloc[i])) ** 0.25
                    regime_weighting.iloc[i] = 1.8
                elif len(negative_moms) >= 3:  # Strong bearish convergence
                    momentum_convergence.iloc[i] = -(abs(mom_1d.iloc[i]) * abs(mom_3d.iloc[i]) * 
                                                   abs(mom_6d.iloc[i]) * abs(mom_12d.iloc[i])) ** 0.25
                    regime_weighting.iloc[i] = 1.8
                else:  # Mixed signals in strong regime
                    weights = [0.4, 0.3, 0.2, 0.1]
                    weighted_sum = sum(w * m for w, m in zip(weights, moms))
                    momentum_convergence.iloc[i] = weighted_sum
                    regime_weighting.iloc[i] = 1.2
                    
            elif current_regime == 2:  # Two regimes aligned (moderate signal)
                if len(positive_moms) >= 3:
                    momentum_convergence.iloc[i] = (abs(mom_1d.iloc[i]) * abs(mom_3d.iloc[i]) * 
                                                  abs(mom_6d.iloc[i]) * abs(mom_12d.iloc[i])) ** 0.25
                    regime_weighting.iloc[i] = 1.3
                elif len(negative_moms) >= 3:
                    momentum_convergence.iloc[i] = -(abs(mom_1d.iloc[i]) * abs(mom_3d.iloc[i]) * 
                                                   abs(mom_6d.iloc[i]) * abs(mom_12d.iloc[i])) ** 0.25
                    regime_weighting.iloc[i] = 1.3
                else:
                    weights = [0.3, 0.3, 0.2, 0.2]
                    weighted_sum = sum(w * m for w, m in zip(weights, moms))
                    momentum_convergence.iloc[i] = weighted_sum
                    regime_weighting.iloc[i] = 1.0
                    
            else:  # Weak or mixed regimes (conservative signal)
                if len(positive_moms) >= 3:
                    momentum_convergence.iloc[i] = (abs(mom_1d.iloc[i]) * abs(mom_3d.iloc[i]) * 
                                                  abs(mom_6d.iloc[i]) * abs(mom_12d.iloc[i])) ** 0.25
                    regime_weighting.iloc[i] = 0.8
                elif len(negative_moms) >= 3:
                    momentum_convergence.iloc[i] = -(abs(mom_1d.iloc[i]) * abs(mom_3d.iloc[i]) * 
                                                   abs(mom_6d.iloc[i]) * abs(mom_12d.iloc[i])) ** 0.25
                    regime_weighting.iloc[i] = 0.8
                else:
                    weights = [0.2, 0.3, 0.3, 0.2]
                    weighted_sum = sum(w * m for w, m in zip(weights, moms))
                    momentum_convergence.iloc[i] = weighted_sum
                    regime_weighting.iloc[i] = 0.6
    
    # Adaptive volatility scaling with regime sensitivity
    adaptive_volatility = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 12:
            current_vol_regime = vol_regime_short.iloc[i] + vol_regime_long.iloc[i]
            
            if current_vol_regime == 2:  # High volatility in both timeframes
                # Use short-term volatility with higher weight
                adaptive_volatility.iloc[i] = intraday_vol_short.iloc[i] * 0.7 + close_vol_short.iloc[i] * 0.3
            elif current_vol_regime == 1:  # Mixed volatility regime
                # Balanced approach
                adaptive_volatility.iloc[i] = intraday_vol_short.iloc[i] * 0.4 + close_vol_long.iloc[i] * 0.6
            else:  # Low volatility regime
                # Emphasize stability with long-term volatility
                adaptive_volatility.iloc[i] = intraday_vol_long.iloc[i] * 0.3 + close_vol_long.iloc[i] * 0.7
    
    # Price efficiency adjustment based on regime
    efficiency_adjustment = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 12:
            current_regime = regime_alignment.iloc[i]
            if current_regime >= 2:  # Strong regime alignment
                # In strong regimes, recent efficiency matters more
                efficiency_adjustment.iloc[i] = range_efficiency.iloc[i] * 0.7 + range_efficiency_3d.iloc[i] * 0.3
            else:  # Weak regime alignment
                # In weak regimes, use longer-term efficiency for stability
                efficiency_adjustment.iloc[i] = range_efficiency.iloc[i] * 0.4 + range_efficiency_6d.iloc[i] * 0.6
    
    # Final factor: regime-weighted momentum convergence with efficiency adjustment
    # and adaptive volatility scaling
    factor = (momentum_convergence * regime_weighting * efficiency_adjustment / 
              (adaptive_volatility + 1e-7))
    
    return factor
