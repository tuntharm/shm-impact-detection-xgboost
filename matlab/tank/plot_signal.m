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
%Flat
%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\plate\silham_p1_flat\silham_l1_p1_flat_1.txt";
%Tank
filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\tank\stlham_p1_tank_new\stlham_l3_p1_tank_1.txt";
%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\tank\silham_p1_tank_new\silham_l1_p1_tank_1.txt"
%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\srubham_p10_tank\srubham_l1_p10_tank_1.txt"
%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\tank\rubham_p10_tank_new\rubham_l16_p10_tank_1.txt"

%------  Broken Sensor----
%filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\Old\rubham_p10_tank_brokens3\rubham_l1_p10_tank_1.txt"

% Read data from the selected file
[time, sensor_data,force_data] = read_labview_file(filename);
impact_type_match = regexp(filename, '(stl|sil|rub|srub)', 'match');
impact_type = impact_type_match{1};
probe_match = regexp(filename, 'p(\d+)', 'tokens'); % Find "p" followed by a number
probe = str2double(probe_match{1}{1}); % Convert to number
loc_match = regexp(filename, 'l(\d+)', 'tokens');
loc = str2double(loc_match{1}{1}); % Convert to number



% Check the number of sensors


%% Figure for raw vs. filtered signals
num_sensors = size(sensor_data, 2);
num_cols = 3; % Three columns for subplots
num_rows = ceil((num_sensors+1) / num_cols); % Calculate required rows
figure;
tiledlayout(num_rows, num_cols); % Create grid layout
tic
for i = 1:num_sensors
    % Extract raw signal
    raw_signal = sensor_data(:, i);
    
    % Apply bandpass filter
    filtered_signal = lowpass_filter(raw_signal, time);
    ToA(i) = compute_ToA(filtered_signal, time, impact_type);
    
    % Plot raw and filtered signal
    nexttile;
    plot_signal_processing(time, raw_signal, filtered_signal, i, ToA(i))
    grid on;
    %ylim([-1.5,1.5])
end
nexttile;
force_filter = lowpass_filter(force_data,time);
plot(time, force_data, 'r', 'LineWidth', 1, 'DisplayName', 'Raw Signal');
hold on
plot(time, force_filter, 'b', 'LineWidth', 1, 'DisplayName', 'Filtered Signal');
xlabel('Time (s)');
ylabel('Voltage (V)');
title(['Loc ' num2str(loc) ' - Force']);
legend;
grid on;
hold off

toc
%% Apply FFT to each sensor signal
% figure;
% tiledlayout(num_rows, num_cols); % Create grid layout
% 
% for i = 1:num_sensors
%     nexttile;
%     raw_signal = sensor_data(:, i);
%     plot_fft(raw_signal, time, i);
% 
%     filtered_signal = lowpass_filter(raw_signal, time);
%     hold on
%     plot_fft(filtered_signal, time, i,'r');  % Re-check the FFT after filterinng
%     %title(['Loc ' num2str(loc)  'FFT - Sensor ' num2str(i)]);
%     legend('Raw Signal', 'Filtered Signal', 'Location', 'best');
%     hold off
% 
% end

%% Apply Hilbert to each sensor signal



figure;
tiledlayout(num_rows, num_cols); % Create grid layout
for i = 1:num_sensors

    nexttile;
    raw_signal = sensor_data(:, i);
    filtered_signal = bandpass_filter(raw_signal, time);

    plot(time, filtered_signal);
    envelope = abs(hilbert(filtered_signal));
    hold on;
    plot(time, envelope, 'r', 'LineWidth', 1.5); % Overlay the envelope
    hold off;
    legend('Raw Signal', 'Hilbert Envelope');
    title(['Loc ' num2str(loc) 'Hilbert Transform - Sensor ' num2str(i)]);
end
grid on 
xlabel('Time (s)');
ylabel('Voltage (V)');

figure;
tiledlayout(num_rows, num_cols); % Create grid layout
for i = 1:num_sensors
    nexttile;
    raw_signal = sensor_data(:, i);
    filtered_signal = bandpass_filter(raw_signal, time);
    hold on
    plot_hilbert_phase(filtered_signal, time, i);  % Re-check the FFT after filterinng

    hold off

end

%% Figure for scalograms
% for i = 1:num_sensors
%     % Extract raw signal
%     raw_signal = sensor_data(:, i);
%     % Apply bandpass filter
%     filtered_signal = lowpass_filter(raw_signal, time);
%     % Compute sampling frequency
%     fs = 1 / mean(diff(time)); 
%     % Plot scalogram
%     figure;
%     cwt(raw_signal, 'amor', fs);
%     title(['Loc ' num2str(loc) 'Scalogram - Sensor ' num2str(i)]);
% 
% end