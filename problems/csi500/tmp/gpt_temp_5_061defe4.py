import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining momentum, volatility, liquidity, order flow, and breakout persistence patterns.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic components
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['overnight_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['momentum_efficiency'] = data['intraday_momentum'] / data['overnight_momentum'].replace(0, np.nan)
    
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volume and amount EMAs
    data['volume_ema_10'] = data['volume'].ewm(span=10, adjust=False).mean()
    data['volume_ema_15'] = data['volume'].ewm(span=15, adjust=False).mean()
    data['amount_ema_10'] = data['amount'].ewm(span=10, adjust=False).mean()
    
    # Volatility EMAs
    data['vol_ema_3'] = data['true_range'].ewm(span=3, adjust=False).mean()
    data['vol_ema_15'] = data['true_range'].ewm(span=15, adjust=False).mean()
    data['vol_regime'] = data['vol_ema_3'] / data['vol_ema_15'].replace(0, np.nan)
    
    # Typical price and EMA
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['typical_price_ema_8'] = data['typical_price'].ewm(span=8, adjust=False).mean()
    
    # Liquidity components
    data['price_efficiency'] = (data['high'] - data['low']) / data['typical_price'].replace(0, np.nan)
    data['volume_intensity'] = data['volume'] / data['volume_ema_10'].replace(0, np.nan)
    data['amount_liquidity'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Order flow components
    data['net_directional_flow'] = np.sign(data['close'] - data['close'].shift(1)) * data['amount']
    data['flow_magnitude'] = data['amount'] / data['amount_ema_10'].replace(0, np.nan)
    
    # Range expansion
    data['range_expansion'] = data['true_range'] / data['true_range'].ewm(span=10, adjust=False).mean().replace(0, np.nan)
    data['volume_expansion'] = data['volume'] / data['volume_ema_15'].replace(0, np.nan)
    
    # Calculate persistence counters and smoothed values
    for i in range(1, len(data)):
        # Momentum persistence
        mom_dir = 1 if data['intraday_momentum'].iloc[i] > 0 else -1
        prev_mom_dir = 1 if data['intraday_momentum'].iloc[i-1] > 0 else -1 if i > 1 else 0
        mom_persistence = data.get('momentum_persistence', pd.Series(0, index=data.index))
        mom_persistence.iloc[i] = mom_persistence.iloc[i-1] + 1 if mom_dir == prev_mom_dir else 1
        
        # Efficiency threshold breaches
        eff_threshold = data['momentum_efficiency'].iloc[i] > 1.2
        eff_persistence = data.get('efficiency_persistence', pd.Series(0, index=data.index))
        eff_persistence.iloc[i] = eff_persistence.iloc[i-1] + 1 if eff_threshold else 0
        
        # Volatility regime persistence
        vol_threshold = data['vol_regime'].iloc[i] > 1.1
        vol_persistence = data.get('vol_persistence', pd.Series(0, index=data.index))
        vol_persistence.iloc[i] = vol_persistence.iloc[i-1] + 1 if vol_threshold else 0
        
        # Liquidity persistence
        liq_condition = (data['volume_intensity'].iloc[i] > 1.1) & (data['amount_liquidity'].iloc[i] > data['amount_liquidity'].quantile(0.6))
        liq_persistence = data.get('liq_persistence', pd.Series(0, index=data.index))
        liq_persistence.iloc[i] = liq_persistence.iloc[i-1] + 1 if liq_condition else 0
        
        # Order flow persistence
        flow_dir = 1 if data['net_directional_flow'].iloc[i] > 0 else -1
        prev_flow_dir = 1 if data['net_directional_flow'].iloc[i-1] > 0 else -1 if i > 1 else 0
        flow_persistence = data.get('flow_persistence', pd.Series(0, index=data.index))
        flow_persistence.iloc[i] = flow_persistence.iloc[i-1] + 1 if flow_dir == prev_flow_dir else 1
        
        # Range expansion persistence
        range_exp = data['range_expansion'].iloc[i] > 1.1
        range_persistence = data.get('range_persistence', pd.Series(0, index=data.index))
        range_persistence.iloc[i] = range_persistence.iloc[i-1] + 1 if range_exp else 0
        
        # Volume expansion persistence
        vol_exp = data['volume_expansion'].iloc[i] > 1.1
        vol_exp_persistence = data.get('vol_exp_persistence', pd.Series(0, index=data.index))
        vol_exp_persistence.iloc[i] = vol_exp_persistence.iloc[i-1] + 1 if vol_exp else 0
        
        # Store persistence series
        data['momentum_persistence'] = mom_persistence
        data['efficiency_persistence'] = eff_persistence
        data['vol_persistence'] = vol_persistence
        data['liq_persistence'] = liq_persistence
        data['flow_persistence'] = flow_persistence
        data['range_persistence'] = range_persistence
        data['vol_exp_persistence'] = vol_exp_persistence
    
    # Apply exponential smoothing
    lambda_val = 0.9
    data['eff_persistence_smooth'] = data['efficiency_persistence'].ewm(alpha=1-lambda_val, adjust=False).mean()
    data['vol_persistence_smooth'] = data['vol_persistence'].ewm(alpha=1-lambda_val, adjust=False).mean()
    data['liq_persistence_smooth'] = data['liq_persistence'].ewm(alpha=1-lambda_val, adjust=False).mean()
    data['flow_persistence_smooth'] = data['flow_persistence'].ewm(alpha=1-lambda_val, adjust=False).mean()
    data['vol_exp_persistence_smooth'] = data['vol_exp_persistence'].ewm(alpha=1-lambda_val, adjust=False).mean()
    
    # Calculate composite factors
    # Momentum-Vollume persistence
    data['momentum_volume_score'] = (data['momentum_persistence'] * data['volume_intensity'] * 
                                   data['eff_persistence_smooth']).ewm(alpha=0.1, adjust=False).mean()
    
    # Volatility-Volume persistence
    data['volatility_volume_score'] = (data['vol_persistence_smooth'] * data['volume_expansion'] * 
                                     np.sign(data['vol_regime'] - 1)).ewm(alpha=0.1, adjust=False).mean()
    
    # Liquidity reversal
    data['price_deviation'] = (data['close'] - data['typical_price_ema_8']) / data['typical_price_ema_8'].replace(0, np.nan)
    data['liquidity_reversal'] = (data['price_deviation'] * data['liq_persistence_smooth'] * 
                                data['volume_intensity']).ewm(alpha=0.1, adjust=False).mean()
    
    # Order flow momentum
    data['flow_momentum'] = (data['flow_persistence_smooth'] * data['flow_magnitude'] * 
                           np.sign(data['net_directional_flow'])).ewm(alpha=0.1, adjust=False).mean()
    
    # Breakout confidence
    data['breakout_confidence'] = (data['range_persistence'] * data['vol_exp_persistence_smooth'] * 
                                 data['range_expansion']).ewm(alpha=0.1, adjust=False).mean()
    
    # Final composite alpha factor
    result = (0.25 * data['momentum_volume_score'] + 
              0.20 * data['volatility_volume_score'] + 
              0.20 * data['liquidity_reversal'] + 
              0.20 * data['flow_momentum'] + 
              0.15 * data['breakout_confidence'])
    
    # Apply final exponential smoothing
    result = result.ewm(alpha=0.05, adjust=False).mean()
    
    return result
