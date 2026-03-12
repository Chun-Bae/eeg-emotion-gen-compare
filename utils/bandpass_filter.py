from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    주어진 EEG 시계열에 대해 Butterworth 밴드패스 필터를 적용합니다.
    
    Parameters:
        data (np.ndarray): 1D EEG 시계열 데이터
        lowcut (float): 하위 컷오프 주파수 (Hz)
        highcut (float): 상위 컷오프 주파수 (Hz)
        fs (int): 샘플링 주파수 (Hz)
        order (int): 필터의 차수

    Returns:
        filtered_data (np.ndarray): 필터링된 EEG 시계열
    """
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data