clear, clc
filename = "data/raw/Old/stlham_p10_flat/stlham_l1_p10_flat_1.txt"
[time, sensor_data, force_data] = read_labview_file_plate(filename);

%% Figure for raw vs. filtered signals% Check the number of sensors
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
    ToA = compute_ToA(filtered_signal, time, 'stl')
    
    % Plot raw and filtered signal
    nexttile;
    plot_signal_processing(time, raw_signal, filtered_signal, i, ToA)
    grid on;
    ylim([-1.5,1.5])
end
nexttile;
force_filter = lowpass_filter(force_data,time);
plot(time, force_data, 'r', 'LineWidth', 1, 'DisplayName', 'Raw Signal');
hold on
plot(time, force_filter, 'b', 'LineWidth', 1, 'DisplayName', 'Filtered Signal');
xlabel('Time (s)');
ylabel('Voltage (V)');
title('Force');
legend;
grid on;
hold off

toc