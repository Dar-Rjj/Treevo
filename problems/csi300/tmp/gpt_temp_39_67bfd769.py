import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['PrevClose'] = data['close'].shift(1)
    data['TR1'] = data['high'] - data['low']
    data['TR2'] = abs(data['high'] - data['PrevClose'])
    data['TR3'] = abs(data['low'] - data['PrevClose'])
    data['TrueRange'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    # Volatility-Adjusted Efficiency
    data['TrueRangeEfficiency'] = (data['close'] - data['open']) / data['TrueRange']
    data['TrueRangeEfficiency'] = data['TrueRangeEfficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Momentum Acceleration Analysis
    data['Return'] = data['close'].pct_change()
    data['Return_5d'] = data['close'].pct_change(5)
    data['Return_10d'] = data['close'].pct_change(10)
    data['Return_3d'] = data['close'].pct_change(3)
    
    data['ShortTermAcceleration'] = (data['Return_5d'] - data['Return_10d']) / 5
    data['UltraShortAcceleration'] = (data['Return_3d'] - data['Return_5d']) / 2
    data['AverageAcceleration'] = (data['ShortTermAcceleration'] + data['UltraShortAcceleration']) / 2
    
    # Efficiency-Weighted Momentum
    data['EfficiencyWeightedMomentum'] = data['TrueRangeEfficiency'] * data['AverageAcceleration']
    
    # Volatility Regime Context
    data['TR_Median_20d'] = data['TrueRange'].rolling(window=20, min_periods=10).median()
    data['VolatilityRegime'] = 1.0  # Normal regime
    data.loc[data['TrueRange'] > 1.2 * data['TR_Median_20d'], 'VolatilityRegime'] = 1.2  # Expansion
    data.loc[data['TrueRange'] < 0.8 * data['TR_Median_20d'], 'VolatilityRegime'] = 0.8  # Contraction
    
    # Asymmetric Volatility Analysis
    data['PrevClose'] = data['close'].shift(1)
    data['ReturnDirection'] = np.where(data['close'] > data['PrevClose'], 1, -1)
    
    # Calculate upside and downside volatility using rolling windows
    upside_returns = []
    downside_returns = []
    
    for i in range(len(data)):
        if i < 20:
            upside_returns.append(np.nan)
            downside_returns.append(np.nan)
            continue
            
        window_returns = data['Return'].iloc[i-19:i+1]
        window_directions = data['ReturnDirection'].iloc[i-19:i+1]
        
        upside_ret = window_returns[window_directions == 1]
        downside_ret = window_returns[window_directions == -1]
        
        upside_vol = upside_ret.std() if len(upside_ret) > 2 else np.nan
        downside_vol = downside_ret.std() if len(downside_ret) > 2 else np.nan
        
        upside_returns.append(upside_vol)
        downside_returns.append(downside_vol)
    
    data['UpsideVolatility'] = upside_returns
    data['DownsideVolatility'] = downside_returns
    data['VolatilityAsymmetry'] = data['UpsideVolatility'] / data['DownsideVolatility']
    data['VolatilityAsymmetry'] = data['VolatilityAsymmetry'].replace([np.inf, -np.inf], np.nan)
    data['VolatilityAsymmetry'] = data['VolatilityAsymmetry'].fillna(1.0)
    
    # Volume-Flow Validation System
    data['Volume_Mean_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['VolumeSurge'] = data['volume'] / data['Volume_Mean_5d']
    
    data['PrevVolume'] = data['volume'].shift(1)
    data['VolumeTrendAlignment'] = np.sign(data['volume'] - data['PrevVolume']) * np.sign(data['close'] - data['PrevClose'])
    data['VolumeConfirmation'] = data['VolumeSurge'] * data['VolumeTrendAlignment']
    
    # Order Flow Quality
    data['EffectiveTurnover'] = data['amount'] / data['volume']
    data['EffectiveTurnover'] = data['EffectiveTurnover'].replace([np.inf, -np.inf], np.nan)
    data['TurnoverVolatility'] = data['EffectiveTurnover'].rolling(window=10, min_periods=5).std()
    
    # Calculate directional consistency (5-day correlation between price change and volume change)
    data['PriceChange'] = data['close'] - data['PrevClose']
    data['VolumeChange'] = data['volume'] - data['PrevVolume']
    
    directional_consistency = []
    for i in range(len(data)):
        if i < 5:
            directional_consistency.append(np.nan)
            continue
            
        price_changes = data['PriceChange'].iloc[i-4:i+1]
        volume_changes = data['VolumeChange'].iloc[i-4:i+1]
        
        if len(price_changes) >= 3 and len(volume_changes) >= 3:
            corr = np.corrcoef(price_changes, volume_changes)[0, 1]
            directional_consistency.append(abs(corr) if not np.isnan(corr) else 0)
        else:
            directional_consistency.append(0)
    
    data['DirectionalConsistency'] = directional_consistency
    data['FlowEfficiency'] = data['DirectionalConsistency'] * data['TurnoverVolatility']
    
    # Range Expansion Component
    data['DailyRange'] = data['high'] - data['low']
    data['AvgRange_5d'] = data['DailyRange'].rolling(window=5, min_periods=3).mean()
    data['RangeExpansion'] = data['DailyRange'] / data['AvgRange_5d']
    
    # Signal Integration and Weighting
    # Core Efficiency-Momentum Factor
    data['BaseComponent'] = data['EfficiencyWeightedMomentum'] * data['VolatilityAsymmetry']
    data['RegimeAdjustment'] = data['TrueRange'] / data['TR_Median_20d']
    data['CoreEfficiencyMomentum'] = data['BaseComponent'] * data['RegimeAdjustment']
    
    # Validation Multiplier
    data['VolumeFlowScore'] = data['VolumeConfirmation'] * data['FlowEfficiency']
    
    # Composite Factor Generation
    data['ValidatedCore'] = data['CoreEfficiencyMomentum'] * data['VolumeFlowScore']
    data['FinalFactor'] = data['ValidatedCore'] * data['RangeExpansion']
    
    # Clean up intermediate columns
    result = data['FinalFactor'].copy()
    
    return result
