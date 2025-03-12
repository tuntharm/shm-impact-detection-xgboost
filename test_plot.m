clear, clc, close all;

% Select a file to analyze
% [file, path] = uigetfile("C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\stlham_p10_flat\stlham_l1_p10_flat_1.txt", 'Select a Raw Data File');
% 
% if isequal(file, 0)
%     disp('No file selected. Exiting.');
%     return;
% end
% filename = fullfile(path, file);

%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\stlham_p10_flat\stlham_l1_p10_flat_1.txt";
% MAC------
%filename = '/Users/tuntharm/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Y4 work/FYP/FYP_Data/Raw_Data/stlham_p1_tank/stlham_l1_p1_tank_1.txt'
%filename = '/Users/tuntharm/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Y4 work/FYP/FYP_Data/Raw_Data/Flat/stlham_p10_flat/stlham_l1_p10_flat_1.txt'
%PC -------
filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\stlham_p1_tank_new\stlham_l1_p1_tank_1.txt";
%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\silham_p1_tank\silham_l1_p1_tank_1.txt"
%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\rubham_p1_tank\rubham_l1_p1_tank_1.txt";
%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\srubham_p10_tank\srubham_l1_p10_tank_1.txt"
[time, sensor_data, ~] = read_labview_file(filename);

raw_signal = sensor_data(:,1);
filtered_signal = lowpass_filter(raw_signal, time);
% %
%% Plot 1 sensor
% plot(time,raw_signal,'b',LineWidth=1)
% xlabel('Time (s)');
% ylabel('Voltage (V)');
% title('Signal from PZT Sensors');
% grid on;
% hold off
% ylim([-1.5,1.5])


%% Plot Different ToA technique

   % ToA = compute_ToA(filtered_signal, time, 'sil');
   %  ToA_aic = compute_ToA(filtered_signal, time, 'rub');
   % 
   %  plot_signal_processing(time, raw_signal, filtered_signal, 1, ToA)
   %  hold on
   %  xline(ToA_aic, '--g', 'DisplayName', 'Time of Arrival (ToA)');
   %  hold off

%% Plot Hilbert Transform 

analytic_signal = hilbert(filtered_signal);
envelope = abs(analytic_signal);

padded_signal = [zeros(100,1); filtered_signal; zeros(100,1)]; % Ensure column-wise padding
envelope = abs(hilbert(padded_signal));
envelope = envelope(101:end-100); % Remove extra padding


figure;
plot(time,filtered_signal,'b', 'LineWidth', 1.5)
hold on
plot(time, envelope, 'r', 'LineWidth', 1.5); % Overlay the envelope
hold off;
legend('Raw Signal', 'Hilbert Envelope');


% plot_hilbert(raw_signal, time,1);  % Re-check the FFT after filterinng
% hold on
% plot_hilbert(filtered_signal, time,1);  % Re-check the FFT after filterinng
%% Plot FFT transform to see main freq
Fs = 2e6; % 2 MHz sampling rate
N = length(filtered_signal);
frequencies = linspace(0, Fs/2, N/2+1);
FFT_signal_raw = abs(fft(raw_signal));
FFT_signal = abs(fft(filtered_signal));
% Convert to dB scale
epsilon = 1e-12;
FFT_signal_dB = 20 * log10(FFT_signal + epsilon);
FFT_signal_dB_raw = 20 * log10(FFT_signal_raw + epsilon);


figure;

plot(frequencies, FFT_signal_dB_raw(1:N/2+1));
hold on
plot(frequencies, FFT_signal_dB(1:N/2+1),'r');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('FFT Analysis of Impact Signal');
grid on;
% 


%% Scalogram CWT
% [cfs, freqs] = cwt(filtered_signal, Fs);
% figure;
% imagesc(time, freqs, abs(cfs));
% axis xy;
% xlabel('Time (s)');
% ylabel('Frequency (Hz)');
% title('Scalogram - Flexural Wave Identification');
% colorbar;

%%
    % 
    % fs = 1 / mean(diff(time)); 
    % % Plot scalogram
    % figure;
    % cwt(raw_signal, 'amor', fs);
    % title('Scalogram - Sensor 1' );
    %     figure;
    % cwt(filtered_signal, 'amor', fs);
    % title('Scalogram - Sensor 1 (filtered)' );
