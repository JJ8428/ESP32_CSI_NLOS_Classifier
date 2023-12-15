import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis


def phase_sanitization(csi_matrix):

    R = np.abs(csi_matrix)
    phase_matrix = np.unwrap(np.angle(csi_matrix), axis=1)
    fit_X = np.arange(0, phase_matrix.shape[1])
    tau_array = []
    phase_mean_array = []
    for m in range(phase_matrix.shape[0]):
        fit_Y = phase_matrix[m]
        tau_t = np.polyfit(fit_X, fit_Y, 1)[0]
        tau_array.append(tau_t)
        old_phase_mean_t = np.mean(phase_matrix[m])
        for n in range(phase_matrix.shape[1]):
            phase_matrix[m, n] -= (((n) * tau_t))
        new_phase_mean_t = np.mean(phase_matrix[m] - old_phase_mean_t)
        phase_mean_array.append(new_phase_mean_t)
    adjusted_csi_matrix = R * np.exp(1j * phase_matrix)

    return adjusted_csi_matrix, phase_matrix, np.mean(tau_array), np.var(tau_array), np.mean(phase_mean_array)


def hampel_filter_correction(csi_matrix, hampel_window_size, conf_thres):

    csi_mag = np.abs(csi_matrix)
    q0 = np.percentile(csi_mag[-1 * (2 * hampel_window_size) + 1:], 15, axis=0)
    q1 = np.percentile(csi_mag[-1 * (2 * hampel_window_size) + 1:], 85, axis=0)
    conf_score = 0
    for i in range(csi_mag.shape[1]):
        if csi_mag[-1 * hampel_window_size][i] >= q0[i] and csi_mag[-1 * hampel_window_size][i] <= q1[i]:
            conf_score += 1
    if conf_score < conf_thres:
        csi_matrix[-1 * hampel_window_size] = csi_matrix[(-1 * hampel_window_size) - 1]
    else:
        pass

    return csi_matrix

def moving_average(data, window_size):

    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def spectral_entropy(signal_freq_domain):

    prob_distribution = signal_freq_domain / np.sum(signal_freq_domain)
    prob_distribution = prob_distribution[prob_distribution > 0]
    entropy = -1 * np.sum(prob_distribution * np.log2(prob_distribution))

    return entropy


def FFT_i_analysis(fft_i_mag_matrix, trim=10, window_size=5):

    fft_i_avg_max_index = 0
    fft_i_avg_peak_height = 0
    fft_i_avg_num_peaks = 0
    fft_i_avg_se = 0

    for i in range(fft_i_mag_matrix.shape[1]):
        fft_i_column = fft_i_mag_matrix[trim:, i]
        smoothed_fft_i = moving_average(fft_i_column, window_size)      
        fft_i_avg_max_index += np.argmax(smoothed_fft_i)
        peaks, properties = find_peaks(fft_i_column, height=np.percentile(fft_i_column, 75))
        fft_i_avg_se = spectral_entropy(fft_i_column)
        
        if len(peaks) > 0:
            fft_i_max_peak_height = np.max(properties['peak_heights'])
            fft_i_avg_peak_height += fft_i_max_peak_height
        fft_i_avg_num_peaks += len(peaks)

    num_columns = fft_i_mag_matrix.shape[1]
    fft_i_avg_max_index /= num_columns # (FE)
    fft_i_avg_peak_height /= num_columns # (FE)
    fft_i_avg_num_peaks /= num_columns # (FE)
    fft_i_avg_se /= num_columns # (FE)

    return fft_i_avg_max_index, fft_i_avg_peak_height, fft_i_avg_num_peaks, fft_i_avg_se


def exponential_decay(t, A, decay_constant):

    # Be aware of frequent overflow errors
    return A * np.exp(-decay_constant * t)


def fit_and_get_decay_constant(row):
    
    indices = np.arange(len(row))
    popt, _ = curve_fit(exponential_decay, indices, row)
    return popt[1]


def CIR_analysis(cir_mag_matrix, trim=10, window_size=5, tolerance=0):
    
    cir_peak = np.max(cir_mag_matrix, axis=1) 
    cir_mean_peak = np.mean(cir_peak) # (FE)
    cir_var_peak = np.var(cir_peak) # (FE)
    cir_var = np.var(cir_mag_matrix[:, trim:-1 * trim], axis=1)
    cir_mean_var = np.mean(cir_var) # (FE)
    cir_rest = np.sum(cir_mag_matrix, axis=1) - cir_peak
    n_factor = np.mean((-1 * cir_peak) / (np.log(cir_peak) * cir_rest)) # (FE)
    
    eps_h = np.apply_along_axis(lambda row: np.trapz(row**2), axis=1, arr=cir_mag_matrix)
    cir_mean_eps_h = np.mean(eps_h) # (FE)
    cir_var_eps_h = np.var(eps_h) # (FE)

    cir_skew = skew(cir_mag_matrix, axis=1)
    cir_mean_skew = np.mean(cir_skew) # (FE)
    cir_var_skew = np.var(cir_skew) # (FE)
    cir_kurtosis = kurtosis(cir_mag_matrix, axis=1)
    cir_mean_kurtosis = np.mean(cir_kurtosis) # (FE)
    cir_var_kurtosis = np.var(cir_kurtosis) # (FE)
    
    fall_times = []
    var_idp = []
    cir_limit = np.percentile(cir_mag_matrix[:, trim:(-1 * trim)], 50, axis=1)
    impulse_array = cir_mag_matrix[::-1].flatten()
    for m in range(cir_mag_matrix.shape[0]):
        fall_peak_index = m * cir_mag_matrix.shape[1]
        fall_flag = 2
        for n in range(cir_mag_matrix.shape[1] - window_size):
            if np.mean(impulse_array[fall_peak_index + n: fall_peak_index + n + window_size]) <= cir_limit[m] + tolerance:
                fall_flag -= 1
                if fall_flag < 0:
                    fall_times.append(n)
                    x_indices = np.arange(n)
                    impulse_decay_profile = impulse_array[fall_peak_index: fall_peak_index + n]
                    impulse_decay_tau = (impulse_decay_profile[0] - impulse_decay_profile[int(n/2)]) / int(n/2) # Polyfit?ww
                    detrended_idp = []
                    for x in x_indices:
                        detrended_idp.append(impulse_decay_profile[x] - (impulse_decay_tau * x))
                    var_idp.append(np.var(detrended_idp))
                    break
    cir_mean_fall_time = np.mean(fall_times) # (FE)
    cir_mean_var_idp = np.mean(var_idp) # (FE)
    cir_var_var_idp = np.var(var_idp) # (FE)

    fft_cir_mag_matrix = np.abs(np.fft.fft(cir_mag_matrix, axis=1))
    fft_cir_mag_matrix = fft_cir_mag_matrix[:, :int(fft_cir_mag_matrix.shape[1]/2)]
    spectral_entropies = np.apply_along_axis(spectral_entropy, axis=1, arr=fft_cir_mag_matrix)
    cir_mean_se = np.mean(spectral_entropies) # (FE)

    t_med = []
    t_rms = []
    for m in range(cir_mag_matrix.shape[0]):
        curr_med = 0
        curr_rms = 0
        for n in range(cir_mag_matrix.shape[1]):
            curr_med += n * cir_mag_matrix[m, n]**2
        curr_med /= eps_h[m]
        for n in range(cir_mag_matrix.shape[1]):
            curr_rms += (n - curr_med)**2 * cir_mag_matrix[m, n]**2
        curr_rms /= eps_h[m]
        t_med.append(curr_med)
        t_rms.append(curr_rms)
    cir_mean_t_med = np.mean(t_med) # (FE)
    cir_var_t_med = np.var(t_med) # (FE)
    cir_mean_t_rms = np.mean(t_rms) # (FE)
    cir_var_t_rms = np.var(t_rms) # (FE)
    
    # Too computationally expensive (+10s per file on a powerful Desktop)
    '''
    decay_constants = np.apply_along_axis(fit_and_get_decay_constant, axis=1, arr=cir_mag_matrix)
    cir_mean_dc = np.mean(decay_constants) # (FE)
    cir_var_dc = np.var(decay_constants) # (FE)
    '''

    correlation_coefficients = []
    for i in range(cir_mag_matrix.shape[0] - 1):
        corr_coef = np.corrcoef(cir_mag_matrix[i], cir_mag_matrix[i + 1])[0, 1]
        correlation_coefficients.append(corr_coef)
    cir_mean_cc = np.mean(correlation_coefficients) # (FE)
    cir_var_cc = np.var(correlation_coefficients) # (FE)
    
    '''
    return cir_mean_peak, cir_var_peak, cir_mean_var, n_factor,\
        cir_mean_eps_h, cir_var_eps_h,\
        cir_mean_skew, cir_var_skew, cir_mean_kurtosis, cir_var_kurtosis,\
        cir_mean_fall_time, cir_mean_var_idp, cir_var_var_idp,\
        cir_mean_t_med, cir_var_t_med, cir_mean_t_rms, cir_var_t_rms,\
        cir_mean_dc, cir_var_dc,\
        cir_mean_cc, cir_var_cc
    '''

    return cir_mean_peak, cir_var_peak, cir_mean_var, n_factor,\
        cir_mean_eps_h, cir_var_eps_h,\
        cir_mean_skew, cir_var_skew, cir_mean_kurtosis, cir_var_kurtosis,\
        cir_mean_fall_time, cir_mean_var_idp, cir_var_var_idp,\
        cir_mean_se,\
        cir_mean_t_med, cir_var_t_med, cir_mean_t_rms, cir_var_t_rms,\
        cir_mean_cc, cir_var_cc


def rssi_analysis(rssi_array):

    return np.mean(rssi_array), np.var(rssi_array)


'''
    https://sci-hub.st/10.1109/icc.2017.7997068
    https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-com.2016.0562#cmu2bf00010-bib-0022
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6960858/#B26-sensors-19-05547
    https://link.springer.com/article/10.1007/s12243-021-00884-6
    https://link.springer.com/article/10.1007/s11277-021-08425-z
    https://www.cs.wm.edu/~yma/files/WiFiSensing_YongsenMa_authorversion.pdf
    https://dl.acm.org/doi/pdf/10.1145/3310194
'''
def feature_extraction(CSI_data_array, zero_padding, rssi_array):

    adjusted_CSI_data_array, _, mean_tau_t, _, mean_phase_t = phase_sanitization(CSI_data_array) # _, _, (FE), _, (FE)

    padded_CSI_data_array = np.array([np.pad(column, (zero_padding, 0), mode='constant', constant_values=0) for column in adjusted_CSI_data_array.T]).T
    adjsuted_nyquist_frequency = 50 + math.ceil(zero_padding/2)
    FFT_i_data = np.fft.fft(np.abs(padded_CSI_data_array), axis=0)[:adjsuted_nyquist_frequency] # FFT for each carrier (nyquist cutoff: 50 Hz)
    CIR_data = np.fft.ifft(adjusted_CSI_data_array, axis=1)

    CSI_mag = np.abs(adjusted_CSI_data_array)
    FFT_i_mag = np.abs(FFT_i_data)
    CIR_mag = np.abs(CIR_data)

    # s_t = skew(CSI_mag, axis=1) # B_2
    # var_t = np.var(CSI_mag, axis=1)
    # k_t = kurtosis(CSI_mag, axis=1) # B_4
    # B_k_t = np.mean((-2*s_t**2 + k_t - s_t * abs(2*s_t**2 - k_t)**.5) / (s_t**2 - k_t)) # (FE)

    csi_i_mean = np.mean(CSI_mag, axis=0)
    csi_i_mean_mean = np.mean(csi_i_mean) # (FE)
    csi_i_var_mean = np.var(csi_i_mean) # (FE)
    csi_i_range_mean = np.max(csi_i_mean) - np.min(csi_i_mean) # (FE)
    csi_i_var = np.var(CSI_mag, axis=0)
    csi_i_mean_var = np.mean(csi_i_var) # (FE)
    csi_i_k_mean = np.mean(np.max(CSI_mag, axis=0)**2 / (2 * np.var(CSI_mag, axis=0))) # (FE)

    fft_i_avg_max_index, fft_i_avg_peak_height, fft_i_avg_num_peaks, fft_i_avg_se = FFT_i_analysis(FFT_i_mag) # (FE)
    cir_mean_peak, cir_var_peak, cir_mean_var, n_factor,\
        cir_mean_eps_h, cir_var_eps_h,\
        cir_mean_skew, cir_var_skew, cir_mean_kurtosis, cir_var_kurtosis,\
        cir_mean_fall_time, cir_mean_var_idp, cir_var_var_idp,\
        cir_mean_se,\
        cir_mean_t_med, cir_var_t_med, cir_mean_t_rms, cir_var_t_rms,\
        cir_mean_cc, cir_var_cc = CIR_analysis(CIR_mag) # All (FE)
    
    # var_phase_i = np.var(adjusted_phase_data_t, axis=0)
    # mean_var_phase_i = np.mean(var_phase_i) # (FE)
    # mean_mean_phase_i = np.mean(np.mean(adjusted_phase_data_t, axis=0)) # (FE)
    # ref_p = cir_mean_peak * np.sum(var_phase_i * csi_i_mean) / np.sum(csi_i_mean)

    mean_rssi, var_rssi = rssi_analysis(rssi_array)

    # Copy and paste this line (excluding return) fe_analysis.py for a historgram of all units
    return mean_tau_t, mean_phase_t,\
        csi_i_mean_mean, csi_i_var_mean, csi_i_range_mean,\
        csi_i_mean_var, csi_i_k_mean,\
        fft_i_avg_max_index, fft_i_avg_peak_height, fft_i_avg_num_peaks, fft_i_avg_se,\
        cir_mean_peak, cir_var_peak, cir_mean_var, n_factor,\
        cir_mean_eps_h, cir_var_eps_h,\
        cir_mean_skew, cir_var_skew, cir_mean_kurtosis, cir_var_kurtosis,\
        cir_mean_fall_time, cir_mean_var_idp, cir_var_var_idp,\
        cir_mean_se,\
        cir_mean_t_med, cir_var_t_med, cir_mean_t_rms, cir_var_t_rms,\
        cir_mean_cc, cir_var_cc,\
        mean_rssi, var_rssi

    # Debug print statement
    '''
    # Print variable names and values
    print(f'mean_tau_t: {mean_tau_t}')
    print(f'mean_phase_t: {mean_phase_t}')
    print(f'csi_i_mean_mean: {csi_i_mean_mean}')
    print(f'csi_i_var_mean: {csi_i_var_mean}')
    print(f'csi_i_range_mean: {csi_i_range_mean}')
    print(f'csi_i_mean_var: {csi_i_mean_var}')
    print(f'csi_i_k_mean: {csi_i_k_mean}')
    # print(f'mean_var_phase_i: {mean_var_phase_i}')
    # print(f'mean_mean_phase_i: {mean_mean_phase_i}')
    # print(f'ref_p: {ref_p}')
    print(f'fft_i_avg_max_index: {fft_i_avg_max_index}')
    print(f'fft_i_avg_peak_height: {fft_i_avg_peak_height}')
    print(f'fft_i_avg_num_peaks: {fft_i_avg_num_peaks}')
    print(f'fft_i_avg_se: {fft_i_avg_se}')
    print(f'cir_mean_peak: {cir_mean_peak}')
    print(f'cir_var_peak: {cir_var_peak}')
    print(f'cir_mean_var: {cir_mean_var}')
    print(f'n_factor: {n_factor}')
    print(f'cir_mean_eps_h: {cir_mean_eps_h}')
    print(f'cir_var_eps_h: {cir_var_eps_h}')
    print(f'cir_mean_skew: {cir_mean_skew}')
    print(f'cir_var_skew: {cir_var_skew}')
    print(f'cir_mean_kurtosis: {cir_mean_kurtosis}')
    print(f'cir_var_kurtosis: {cir_var_kurtosis}')
    print(f'cir_mean_fall_time: {cir_mean_fall_time}')
    print(f'cir_mean_var_idp: {cir_mean_var_idp}')
    print(f'cir_var_var_idp: {cir_var_var_idp}')
    print(f'cir_mean_se: {cir_mean_se}')
    print(f'cir_mean_t_med: {cir_mean_t_med}')
    print(f'cir_var_t_med: {cir_var_t_med}')
    print(f'cir_mean_t_rms: {cir_mean_t_rms}')
    print(f'cir_var_t_rms: {cir_var_t_rms}')
    print(f'cir_mean_dc: {cir_mean_dc}')
    print(f'cir_var_dc: {cir_var_dc}')
    print(f'cir_mean_cc: {cir_mean_cc}')
    print(f'cir_var_cc: {cir_var_cc}')
    print(f'mean_rssi: {mean_rssi}')
    print(f'var_rssi: {var_rssi}')
    '''
