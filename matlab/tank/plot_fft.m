function plot_fft(signal, time, sensor_id, color)
    if nargin < 4
        color = 'b';  % Default color is blue if not provided
    end
    
    fs = 1 / mean(diff(time));       % Sampling frequency
    N = length(signal);              % Number of samples
    f = linspace(0, fs/2, floor(N/2) + 1);  % Frequency vector (single-sided)

    % Remove DC Offset
    signal = signal - mean(signal);  % Center the signal around zero

    % Apply Hamming Window to reduce spectral leakage
    window = hamming(N);
    signal_windowed = signal .* window;

    % Apply FFT
    Y = fft(signal_windowed);
    P2 = abs(Y / N);                 % Two-sided spectrum
    P1 = P2(1:floor(N/2) + 1);       % Single-sided spectrum
    P1(2:end-1) = 2 * P1(2:end-1);   % Adjust amplitude for single-sided

    % Avoid log(0) by adding a small epsilon
    epsilon = 1e-12;  
    P1_dB = 20 * log10(P1 + epsilon);  % Convert to dB scale

    % Plotting
    plot(f, P1_dB, 'Color', color, 'LineWidth', 0.5);
    title(['FFT Spectrum - Sensor ' num2str(sensor_id)]);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    xlim([0, 100000]);  % Limit to Nyquist frequency
end

