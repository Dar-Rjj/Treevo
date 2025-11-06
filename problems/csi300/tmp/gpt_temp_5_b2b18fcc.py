import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Price Fractal Analysis
    # Fractal Dimension Estimation
    data['Intraday_Range_Fractal'] = np.log(data['high'] - data['low'] + 1e-8) / np.log(data['volume'] + 1)
    data['Multi_day_Fractal_Consistency'] = data['Intraday_Range_Fractal'].rolling(window=5).std()
    data['Fractal_Stability_Score'] = 1 / (1 + data['Multi_day_Fractal_Consistency'])
    
    # Price Path Complexity
    data['Directional_Changes'] = (data['close'] != data['close'].shift(1)).rolling(window=5).sum()
    data['Path_Efficiency'] = (data['close'] - data['close'].shift(4)) / (
        abs(data['close'] - data['close'].shift(1)).rolling(window=4).sum() + 1e-8)
    data['Complexity_Adjusted_Return'] = (data['close'] / data['close'].shift(5) - 1) * data['Path_Efficiency']
    
    # Volatility Fractal Signature
    data['Volatility_Clustering_Pattern'] = (data['high'] - data['low']).rolling(window=10).apply(
        lambda x: x.autocorr(lag=1), raw=False)
    data['Fractal_Volatility_Ratio'] = (data['high'] - data['low']) / (abs(data['close'] - data['close'].shift(1)) + 1e-8)
    data['Volatility_Fractal_Score'] = data['Volatility_Clustering_Pattern'] * data['Fractal_Volatility_Ratio']
    
    # Volume Microstructure Regime Detection
    # Volume Fractal Analysis
    data['Volume_Scaling_Exponent'] = np.log(data['volume'] + 1) / np.log(data['amount'] + 1e-8)
    data['Volume_Clustering_Intensity'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['Microstructure_Noise'] = data['Volume_Scaling_Exponent'].rolling(window=5).std()
    
    # Trade Size Distribution
    data['Large_Trade_Dominance'] = data['amount'] / (data['volume'] * data['close'] + 1e-8)
    data['Trade_Size_Skewness'] = (data['amount'] / (data['volume'] + 1e-8)) / (
        (data['amount'] / (data['volume'] + 1e-8)).rolling(window=5).mean())
    data['Distribution_Consistency'] = 1 / (1 + abs(data['Trade_Size_Skewness'] - 1))
    
    # Volume-Price Co-Fractality
    data['Volume_Range_Correlation'] = data['volume'].rolling(window=5).corr(data['high'] - data['low'])
    data['Co_Fractal_Strength'] = data['Volume_Range_Correlation'] * data['Fractal_Stability_Score']
    data['Microstructure_Regime'] = np.sign(data['Co_Fractal_Strength']) * data['Microstructure_Noise']
    
    # Regime-Switching Momentum Dynamics
    # Fractal-Regime Identification
    data['High_Complexity_Regime'] = (data['Fractal_Stability_Score'] < 0.5) & (data['Microstructure_Noise'] > 0.1)
    data['Low_Complexity_Regime'] = (data['Fractal_Stability_Score'] > 0.8) & (data['Volume_Clustering_Intensity'] < 1.2)
    data['Transition_Regime'] = ~(data['High_Complexity_Regime'] | data['Low_Complexity_Regime'])
    
    # Regime-Adaptive Momentum
    data['High_Complexity_Momentum'] = data['Complexity_Adjusted_Return'] * data['Volatility_Fractal_Score']
    data['Low_Complexity_Momentum'] = (data['close'] / data['close'].shift(10) - 1) * data['Distribution_Consistency']
    data['Transition_Momentum'] = (data['close'] / data['close'].shift(5) - 1) * data['Co_Fractal_Strength']
    
    # Regime Persistence Signals
    def consecutive_count(series):
        return series.groupby((series != series.shift()).cumsum()).cumcount() + 1
    
    data['Regime_Duration'] = np.where(
        data['High_Complexity_Regime'], 
        consecutive_count(data['High_Complexity_Regime']),
        np.where(
            data['Low_Complexity_Regime'],
            consecutive_count(data['Low_Complexity_Regime']),
            consecutive_count(data['Transition_Regime'])
        )
    )
    
    data['Regime_Strength'] = data['Regime_Duration'] * abs(data['Microstructure_Regime'])
    
    data['Regime_Weighted_Momentum'] = np.where(
        data['High_Complexity_Regime'], data['High_Complexity_Momentum'],
        np.where(
            data['Low_Complexity_Regime'], data['Low_Complexity_Momentum'],
            data['Transition_Momentum']
        )
    ) * data['Regime_Strength']
    
    # Price-Volume Fractal Divergence
    # Fractal Dimension Divergence
    def rolling_slope(series, window):
        def calc_slope(x):
            if len(x) < 2:
                return np.nan
            return np.polyfit(range(len(x)), x, 1)[0]
        return series.rolling(window=window).apply(calc_slope, raw=True)
    
    data['Price_Fractal_Trend'] = rolling_slope(data['Intraday_Range_Fractal'], 5)
    data['Volume_Fractal_Trend'] = rolling_slope(data['Volume_Scaling_Exponent'], 5)
    data['Fractal_Divergence'] = data['Price_Fractal_Trend'] - data['Volume_Fractal_Trend']
    
    # Microstructure-Momentum Alignment
    data['Noise_Adjusted_Return'] = (data['close'] / data['close'].shift(3) - 1) / (1 + data['Microstructure_Noise'])
    data['Alignment_Score'] = np.sign(data['Noise_Adjusted_Return']) * np.sign(data['Microstructure_Regime'])
    data['Divergence_Strength'] = abs(data['Fractal_Divergence']) * data['Alignment_Score']
    
    # Fractal Breakout Detection
    data['Fractal_Breakout'] = data['Fractal_Divergence'] > data['Fractal_Divergence'].rolling(window=10).std()
    data['Breakout_Momentum'] = data['Regime_Weighted_Momentum'] * data['Fractal_Divergence']
    data['Validated_Breakout'] = data['Breakout_Momentum'] * data['Divergence_Strength']
    
    # Multi-Fractal Risk Adjustment
    # Fractal Risk Components
    data['Path_Risk'] = 1 / (data['Path_Efficiency'] + 1e-8)
    data['Volume_Risk'] = 1 / (data['Distribution_Consistency'] + 1e-8)
    data['Regime_Risk'] = 1 / (1 + data['Regime_Strength'])
    
    # Combined Risk Measure
    data['Fractal_Risk_Score'] = data['Path_Risk'] * data['Volume_Risk'] * data['Regime_Risk']
    data['Risk_Adjusted_Momentum'] = data['Regime_Weighted_Momentum'] / (1 + data['Fractal_Risk_Score'])
    data['Risk_Weighted_Breakout'] = data['Validated_Breakout'] / (1 + data['Fractal_Risk_Score'])
    
    # Dynamic Risk Thresholds
    data['High_Risk_Regime'] = data['Fractal_Risk_Score'] > data['Fractal_Risk_Score'].rolling(window=10).mean()
    data['Risk_Adjusted_Signal'] = data['Risk_Weighted_Breakout'] * (1 - 0.5 * data['High_Risk_Regime'])
    data['Fractal_Risk_Premium'] = data['Risk_Adjusted_Signal'] * data['Fractal_Risk_Score']
    
    # Adaptive Alpha Construction
    # Core Fractal Factors
    data['Fractal_Momentum_Core'] = data['Risk_Adjusted_Momentum'] * data['Fractal_Stability_Score']
    data['Breakout_Core'] = data['Risk_Adjusted_Signal'] * data['Co_Fractal_Strength']
    data['Regime_Core'] = data['Microstructure_Regime'] * data['Regime_Weighted_Momentum']
    
    # Multi-Scale Integration
    data['Short_term_Component'] = data['Fractal_Momentum_Core'] * data['Volume_Clustering_Intensity']
    data['Medium_term_Component'] = data['Breakout_Core'] * data['Regime_Strength']
    data['Long_term_Component'] = data['Regime_Core'] * data['Fractal_Risk_Premium']
    
    # Final Alpha Signal
    data['Scale_Weighted_Average'] = (
        data['Short_term_Component'] * 0.4 + 
        data['Medium_term_Component'] * 0.35 + 
        data['Long_term_Component'] * 0.25
    )
    data['Fractal_Validation'] = data['Scale_Weighted_Average'] * data['Alignment_Score']
    data['Predictive_Alpha'] = data['Fractal_Validation'] * (1 + data['Fractal_Divergence'])
    
    return data['Predictive_Alpha']
