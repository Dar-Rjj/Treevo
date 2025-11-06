import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum Asymmetry factor combining directional pressure, 
    volatility regimes, price gaps, amount-volume divergence, and multi-timeframe momentum.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Required minimum data points for calculations
    min_periods = 10
    
    for i in range(len(data)):
        if i < min_periods:
            factor.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        # Get historical data safely
        def safe_get(idx, col):
            if idx < 0:
                return np.nan
            return data.iloc[idx][col]
        
        # Directional Pressure Dynamics
        # Bull-bear pressure
        if current['close'] != current['low']:
            bull_bear = ((current['high'] - current['close']) / 
                        (current['close'] - current['low']) * current['volume'])
        else:
            bull_bear = 0
            
        # Opening momentum
        prev_close = safe_get(i-1, 'close')
        prev_volume = safe_get(i-1, 'volume')
        if prev_close != 0 and prev_volume != 0:
            open_momentum = ((current['open'] - prev_close) / prev_close * 
                           current['volume'] / prev_volume)
        else:
            open_momentum = 0
            
        # Closing pressure
        if current['high'] != current['low']:
            close_pressure = ((current['close'] - current['open']) / 
                            (current['high'] - current['low']) * current['amount'])
        else:
            close_pressure = 0
            
        directional_pressure = bull_bear + open_momentum + close_pressure
        
        # Volatility Regime Analysis
        # Intraday volatility efficiency
        if current['high'] != current['low']:
            intraday_vol_eff = (abs(current['close'] - current['open']) / 
                              (current['high'] - current['low']))
        else:
            intraday_vol_eff = 0
            
        # Calculate rolling standard deviations
        close_prices = [safe_get(j, 'close') for j in range(i-4, i+1)]
        close_prices_prev = [safe_get(j, 'close') for j in range(i-9, i-4)]
        
        if len([x for x in close_prices if not np.isnan(x)]) >= 3:
            std_current = np.std([x for x in close_prices if not np.isnan(x)])
        else:
            std_current = 0
            
        if len([x for x in close_prices_prev if not np.isnan(x)]) >= 3:
            std_prev = np.std([x for x in close_prices_prev if not np.isnan(x)])
        else:
            std_prev = 0
            
        # Regime persistence
        if std_prev != 0:
            regime_persistence = (std_current - std_prev) / std_prev
        else:
            regime_persistence = 0
            
        # Volume-volatility coupling
        if prev_close != 0:
            vol_vol_coupling = (current['volume'] * (current['high'] - current['low']) / prev_close)
        else:
            vol_vol_coupling = 0
            
        volatility_regime = intraday_vol_eff * std_current + regime_persistence + vol_vol_coupling
        
        # Price Gap Dynamics
        # Gap absorption
        gap = abs(current['open'] - prev_close)
        if gap != 0:
            gap_absorption = ((current['close'] - current['open']) / gap * 
                            np.sign(current['close'] - current['open']))
        else:
            gap_absorption = 0
            
        # Gap momentum
        if prev_close != 0:
            gap_momentum = ((current['open'] - prev_close) * 
                          (current['close'] - prev_close) / prev_close)
        else:
            gap_momentum = 0
            
        # Gap reversal probability
        gap_reversal = (np.sign(current['open'] - prev_close) * 
                       np.sign(current['close'] - current['open']))
        
        price_gap = gap_absorption + gap_momentum + gap_reversal
        
        # Amount-Volume Divergence
        # Large trade pressure
        large_trade_pressure = (current['amount'] / current['volume'] * 
                              (current['close'] - prev_close)) if current['volume'] != 0 else 0
        
        # Volume-amount efficiency
        prev_amount = safe_get(i-1, 'amount')
        if prev_amount is not None and current['amount'] != prev_amount:
            vol_amount_eff = ((current['volume'] - prev_volume) / 
                            (current['amount'] - prev_amount) * current['close'])
        else:
            vol_amount_eff = 0
            
        # Institutional pressure (using 5-day moving averages)
        amount_ma_5 = np.mean([safe_get(j, 'amount') for j in range(i-4, i+1) 
                              if not np.isnan(safe_get(j, 'amount'))])
        volume_ma_5 = np.mean([safe_get(j, 'volume') for j in range(i-4, i+1) 
                              if not np.isnan(safe_get(j, 'volume'))])
        
        if amount_ma_5 != 0 and volume_ma_5 != 0:
            institutional_pressure = ((current['amount'] / amount_ma_5) * 
                                    (current['volume'] / volume_ma_5))
        else:
            institutional_pressure = 0
            
        amount_volume_div = large_trade_pressure + vol_amount_eff + institutional_pressure
        
        # Multi-Timeframe Momentum
        # Short-term acceleration
        close_t2 = safe_get(i-2, 'close')
        close_t1 = safe_get(i-1, 'close')
        close_t3 = safe_get(i-3, 'close')
        
        if (close_t2 is not None and close_t1 is not None and 
            close_t3 is not None and prev_close is not None):
            short_term_accel = ((current['close'] - close_t2) - 
                              2 * (close_t1 - close_t3))
        else:
            short_term_accel = 0
            
        # Medium-term momentum stability (correlation approximation)
        recent_prices = [safe_get(j, 'close') for j in range(i-4, i+1) 
                        if not np.isnan(safe_get(j, 'close'))]
        prev_prices = [safe_get(j, 'close') for j in range(i-9, i-4) 
                      if not np.isnan(safe_get(j, 'close'))]
        
        if len(recent_prices) >= 3 and len(prev_prices) >= 3:
            # Simple correlation approximation using price differences
            recent_changes = [recent_prices[k] - recent_prices[k-1] 
                            for k in range(1, len(recent_prices))]
            prev_changes = [prev_prices[k] - prev_prices[k-1] 
                          for k in range(1, len(prev_prices))]
            
            if len(recent_changes) >= 2 and len(prev_changes) >= 2:
                corr_approx = np.corrcoef(recent_changes, prev_changes[:len(recent_changes)])[0,1]
                if np.isnan(corr_approx):
                    corr_approx = 0
            else:
                corr_approx = 0
                
            long_term_std = np.std([safe_get(j, 'close') for j in range(i-9, i+1) 
                                  if not np.isnan(safe_get(j, 'close'))])
            momentum_stability = corr_approx * long_term_std
        else:
            momentum_stability = 0
            
        # Momentum regime shift
        close_t4 = safe_get(i-4, 'close')
        close_t9 = safe_get(i-9, 'close')
        
        if (close_t4 is not None and close_t9 is not None and 
            prev_close is not None and current['close'] is not None):
            momentum_shift = (np.sign(current['close'] - close_t4) * 
                            np.sign(close_t4 - close_t9))
        else:
            momentum_shift = 0
            
        multi_timeframe = short_term_accel + momentum_stability + momentum_shift
        
        # Combine all components with equal weighting
        factor.iloc[i] = (directional_pressure + volatility_regime + 
                         price_gap + amount_volume_div + multi_timeframe)
    
    # Normalize the factor
    if len(factor) > 0:
        factor = (factor - factor.mean()) / (factor.std() + 1e-8)
    
    return factor
