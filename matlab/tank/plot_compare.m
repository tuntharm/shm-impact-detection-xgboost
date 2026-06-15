clear, clc, close all;

% Define filenames
% filenames = {
%     "data/raw/stlham_p1_tank/stlham_l6_p1_tank_1.txt",
%     "data/raw/silham_p1_tank/silham_l6_p1_tank_1.txt",
%     "data/raw/rubham_p10_tank/rubham_l6_p10_tank_1.txt",
%     "data/raw/srubham_p10_tank/srubham_l6_p10_tank_1.txt"
% };


%% Sensor 1 vs Sensor 2
% figure;
% tiledlayout(2,2);
% 
% % Loop through each file
% for i = 1:length(filenames)
%     filename = filenames{i};
% 
%     % Read data
%     [time, sensor_data, force_data] = read_labview_file(filename);
% 
%     % Extract impact type and probe number
%     impact_type_match = regexp(filename, '(stl|sil|rub|srub)', 'match');
%     impact_type = impact_type_match{1};
%     probe_match = regexp(filename, 'p(\d+)', 'tokens'); % Find "p" followed by a number
%     probe = str2double(probe_match{1}{1}); % Convert to number
% 
%     % Check the number of sensors
%     num_sensors = size(sensor_data, 2);
% 
%     % Extract and filter raw signal from first sensor
%     raw_signal_1 = sensor_data(:,1);
%     filtered_signal_1 = lowpass_filter(raw_signal_1, time);
%     raw_signal_2 = sensor_data(:,2);
%     filtered_signal_2 = lowpass_filter(raw_signal_2, time);
% 
%     % Plot raw and filtered signal
%     nexttile;
%     plot(time, filtered_signal_1, 'b', 'DisplayName', 'Sensor 1'); hold on;
%     plot(time, filtered_signal_2, 'r', 'DisplayName', 'Sensor 2');
% 
%     xlabel('Time (s)');
%     ylabel('Voltage (V)');
%     title(['Loc 6 - ', impact_type]);
%     legend;
%     grid on;
% end

% %% Plot Envelope
% %%Create tiled layout for subplots
% figure;
% tiledlayout(2,2);
% 
% %Loop through each file
% for i = 1:length(filenames)
%     filename = filenames{i};
% 
%     %Read data
%     [time, sensor_data, force_data] = read_labview_file(filename);
% 
%     %Extract impact type and probe number
%     impact_type_match = regexp(filename, '(stl|sil|rub|srub)', 'match');
%     impact_type = impact_type_match{1};
%     probe_match = regexp(filename, 'p(\d+)', 'tokens'); % Find "p" followed by a number
%     probe = str2double(probe_match{1}{1}); % Convert to number
% 
%     %Check the number of sensors
%     num_sensors = size(sensor_data, 2);
% 
%     %Extract and filter raw signal from first sensor
%     raw_signal_1 = sensor_data(:,1);
%     filtered_signal_1 = lowpass_filter(raw_signal_1, time);
% 
%     analytic_signal = hilbert(filtered_signal_1);
%     envelope = abs(analytic_signal);
% 
%    % Plot raw and filtered signal
%     nexttile;
% 
%     plot(time,filtered_signal_1,'r', 'LineWidth', 1,'DisplayName','Filtered Signal' )
%      hold on
%      plot(time,envelope,'b', 'LineWidth', 1,'DisplayName','Hilbert Transform')
%     xlabel('Time (s)');
%     ylabel('Voltage (V)');
%     title(['Loc 1 - Sensor 1 - ', impact_type]);
%     legend;
%     grid on;
% end



%% Plot Compare ToA

filenames = {
    "data/raw/stlham_p1_tank_new/stlham_l1_p1_tank_1.txt",
    "data/raw/stlham_p1_tank_new/stlham_l2_p1_tank_1.txt",
    "data/raw/stlham_p1_tank_new/stlham_l3_p1_tank_1.txt",
    "data/raw/stlham_p1_tank_new/stlham_l4_p1_tank_1.txt",

};


%%Create tiled layout for subplots
figure;
tiledlayout(2,2);

%Loop through each file
for i = 1:length(filenames)
    filename = filenames{i};

    %Read data
    [time, sensor_data, force_data] = read_labview_file(filename);

    %Extract impact type and probe number
    impact_type_match = regexp(filename, '(stl|sil|rub|srub)', 'match');
    impact_type = impact_type_match{1};
    probe_match = regexp(filename, 'p(\d+)', 'tokens'); % Find "p" followed by a number
    probe = str2double(probe_match{1}{1}); % Convert to number

    %Check the number of sensors
    num_sensors = size(sensor_data, 2);

    %Extract and filter raw signal from first sensor
    raw_signal_1 = sensor_data(:,1);
    filtered_signal_1 = lowpass_filter(raw_signal_1, time);
    ToA = compute_ToA(filtered_signal_1, time, impact_type);

   % Plot raw and filtered signal
    nexttile;

    plot(time,filtered_signal_1, 'LineWidth', 0.5,'DisplayName','Filtered Signal' )
     hold on
    xline(ToA, '--r', 'DisplayName', 'ToA');

    xlabel('Time (s)');
    ylabel('Voltage (V)');
    title(['ToA = ',num2str(ToA)]);
    legend;
    grid on;
end