import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Multi-Scale Momentum with Volatility-Weighted Efficiency
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Absorption
    # Range-based momentum calculation
    data['short_range_momentum'] = (data['high'] - data['low']).rolling(window=3).mean() / (data['high'] - data['low'])
    data['medium_range_momentum'] = (data['high'] - data['low']).rolling(window=8).mean() / (data['high'] - data['low'])
    data['range_momentum_divergence'] = data['short_range_momentum'] - data['medium_range_momentum']
    
    # Directional consistency analysis
    data['price_direction'] = np.where(data['close'] > data['close'].shift(1), 1, -1)
    data['direction_consistency'] = data['price_direction'].rolling(window=5).sum() / 5
    
    # Liquidity absorption dynamics
    data['volume_concentration'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + 
                                                    data['volume'].shift(3) + data['volume'].shift(4) + data['volume'].shift(5))
    data['price_level_absorption'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['absorption_efficiency'] = (abs(data['close'] - data['open']) * data['volume'] / data['amount']).replace(0, np.nan)
    
    # Absorption trend persistence
    data['absorption_trend'] = data['absorption_efficiency'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if not x.isna().any() else np.nan
    )
    
    # Asymmetric momentum-absorption regime detection
    data['momentum_continuation'] = np.where(
        (data['short_range_momentum'] > data['medium_range_momentum']) & 
        (data['absorption_efficiency'] < data['absorption_efficiency'].rolling(window=10).mean()),
        1, 0
    )
    data['accumulation_phase'] = np.where(
        (data['short_range_momentum'] < data['medium_range_momentum']) & 
        (data['absorption_efficiency'] > data['absorption_efficiency'].rolling(window=10).mean()),
        1, 0
    )
    
    # Gap Momentum with Volatility-Adjusted Pressure
    # Gap momentum characteristics
    data['raw_gap'] = (data['open'] / data['close'].shift(1)) - 1
    data['gap_sustainability'] = (data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    data['gap_fill_analysis'] = (np.minimum(data['open'], data['close']) / np.maximum(data['open'], data['close'])) - 1
    
    # Gap momentum persistence
    data['gap_persistence'] = data['raw_gap'].rolling(window=3).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if not x.isna().any() else np.nan
    )
    
    # Volatility-adjusted pressure accumulation
    data['intraday_buying_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)) * data['volume']
    data['intraday_selling_pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)) * data['volume']
    data['net_pressure_accumulation'] = (
        data['intraday_buying_pressure'].rolling(window=5).sum() - 
        data['intraday_selling_pressure'].rolling(window=5).sum()
    )
    
    # Gap-pressure volatility regime analysis
    data['trend_initiation'] = np.where(
        (abs(data['raw_gap']) > data['raw_gap'].rolling(window=20).std()) & 
        (data['net_pressure_accumulation'] > 0),
        1, 0
    )
    data['false_breakout'] = np.where(
        (abs(data['raw_gap']) > data['raw_gap'].rolling(window=20).std()) & 
        (data['gap_sustainability'] < 0.5),
        1, 0
    )
    
    # Volatility-Weighted Elasticity with Movement Efficiency
    # Multi-period price elasticity
    data['daily_elasticity'] = (abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) / 
                               (abs(data['high'] - data['low']) / data['close'])).replace(0, np.nan)
    
    # Multi-period elasticity
    high_3d = data['high'].rolling(window=3).max()
    low_3d = data['low'].rolling(window=3).min()
    data['multi_period_elasticity'] = (
        abs((data['close'] - data['close'].shift(3)) / data['close'].shift(3)) / 
        ((high_3d - low_3d) / data['close'])
    ).replace(0, np.nan)
    
    # Elasticity trend
    data['elasticity_trend'] = (
        data['daily_elasticity'].rolling(window=3).mean() - 
        data['daily_elasticity'].rolling(window=10).mean()
    )
    
    # Volatility-adjusted movement efficiency
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['movement_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['true_range'].replace(0, np.nan)
    
    # Efficiency trend
    data['efficiency_trend'] = (
        data['movement_efficiency'].rolling(window=3).mean() - 
        data['movement_efficiency'].rolling(window=10).mean()
    )
    
    # Asymmetric volatility-weighted efficiency confirmation
    data['volatile_breakout'] = np.where(
        (data['daily_elasticity'] > data['daily_elasticity'].rolling(window=20).mean()) & 
        (data['movement_efficiency'] > data['movement_efficiency'].rolling(window=20).mean()),
        1, 0
    )
    data['consolidation_phase'] = np.where(
        (data['daily_elasticity'] < data['daily_elasticity'].rolling(window=20).mean()) & 
        (data['movement_efficiency'] < data['movement_efficiency'].rolling(window=20).mean()),
        1, 0
    )
    
    # Asymmetric Volatility-Regime Adaptive Integration
    # Multi-timeframe volatility regime detection
    data['volatility_regime'] = (data['high'] - data['low']).rolling(window=20).mean()
    volatility_quantiles = data['volatility_regime'].quantile([0.33, 0.67])
    
    # Volume-volatility regime confirmation
    data['volume_regime'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Volume-volatility correlation
    data['volume_volatility_corr'] = data['volume'].rolling(window=15).corr(data['high'] - data['low'])
    
    # Asymmetric volatility-adaptive signal generation
    # High volatility regime
    high_vol_mask = data['volatility_regime'] > volatility_quantiles[0.67]
    # Low volatility regime  
    low_vol_mask = data['volatility_regime'] < volatility_quantiles[0.33]
    # Medium volatility regime
    medium_vol_mask = ~high_vol_mask & ~low_vol_mask
    
    # Initialize final factor
    data['final_factor'] = 0
    
    # High volatility: emphasize elasticity-efficiency
    data.loc[high_vol_mask, 'final_factor'] = (
        0.4 * data['elasticity_trend'] + 
        0.3 * data['efficiency_trend'] + 
        0.2 * data['range_momentum_divergence'] + 
        0.1 * data['net_pressure_accumulation'].rank(pct=True)
    )
    
    # Low volatility: emphasize momentum absorption
    data.loc[low_vol_mask, 'final_factor'] = (
        0.4 * data['range_momentum_divergence'] + 
        0.3 * data['absorption_efficiency'].rank(pct=True) + 
        0.2 * data['gap_sustainability'] + 
        0.1 * data['elasticity_trend']
    )
    
    # Medium volatility: balanced approach
    data.loc[medium_vol_mask, 'final_factor'] = (
        0.25 * data['range_momentum_divergence'] + 
        0.25 * data['elasticity_trend'] + 
        0.25 * data['gap_sustainability'] + 
        0.15 * data['absorption_efficiency'].rank(pct=True) + 
        0.1 * data['net_pressure_accumulation'].rank(pct=True)
    )
    
    # Apply volatility scaling
    volatility_scaling = 1 / (1 + data['volatility_regime'].rank(pct=True))
    data['final_factor'] = data['final_factor'] * volatility_scaling
    
    # Directional adjustment based on momentum confirmation
    momentum_direction = np.sign(data['range_momentum_divergence'] + data['net_pressure_accumulation'].rank(pct=True))
    data['final_factor'] = data['final_factor'] * momentum_direction
    
    return data['final_factor']
