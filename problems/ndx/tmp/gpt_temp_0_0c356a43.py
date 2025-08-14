import numpy as np

def heuristics_v2(df):
    high_low_weight = 1.0
    smoothing_period = 10
    
    high_low_diff = (df['high'] - df['low']) * high_low_weight
    fft_result = np.fft.rfft(high_low_diff)
    amplitudes = np.abs(fft_result)
    max_amplitude_index = np.argmax(amplitudes)
    selected_amplitude = np.real(np.fft.irfft(fft_result[max_amplitude_index] * (fft_result == fft_result[max_amplitude_index])))
    heuristics_matrix = pd.Series(selected_amplitude).rolling(window=smoothing_period).mean().dropna()
    
    return heuristics_matrix
