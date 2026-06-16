clear, clc 
filename = "data/raw/Old/silham_l1_p1_tank_noise.txt"

    % Read the main sensor data file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    % Skip header lines until the second occurrence of '***End_of_Header***'
    header_count = 0;
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, '***End_of_Header***')
            header_count = header_count + 1;
        end
        if header_count == 2
            break;
        end
    end

    % Read column names to detect the number of columns
    column_names = strsplit(fgetl(fid), '\t');
    num_columns = length(column_names)-1; 

    % Generate formatSpec dynamically
    formatSpec = repmat('%f ', 1, num_columns);
    formatSpec = strtrim(formatSpec);

    % Read the sensor data
    data = textscan(fid, formatSpec, 'Delimiter', '\t');
    fclose(fid);

    % Extract time and sensor readings
    time = data{1} - min(data{1});               % First column is time
    %sensor_data = cell2mat(data(2:end-1));         % Remaining columns are sensor dat
    %force_data = cell2mat(data(end));  

    sensor_data = cell2mat(data(2:end));
    % Automatically detect corresponding force data file
    [folder, base_name, ~] = fileparts(filename);
    force_filename = fullfile(folder, strrep(base_name, '_tank_', '_tank_f_') + ".txt");

    if isfile(force_filename)
        % Read the force data
        force_data = read_force_data(force_filename);
    else
        warning(['Force data file not found: ', force_filename]);
        force_data = NaN(size(time)); % Fill with NaN if force data is missing
    end

%% Plot
% Check the number of sensors
num_sensors = size(sensor_data, 2);
num_cols = 3; % Three columns for subplots
num_rows = ceil((num_sensors+1) / num_cols); % Calculate required rows

% figure;
% tiledlayout(num_rows, num_cols); % Create grid layout
% tic
% for i = 1:num_sensors
%     % Extract raw signal
%     raw_signal = sensor_data(:, i);
% 
%     % Apply bandpass filter
%     filtered_signal = lowpass_filter(raw_signal, time);
%     ToA = compute_ToA(filtered_signal, time, 'sil');
% 
%     % Plot raw and filtered signal
%     nexttile;
%     plot_signal_processing(time, raw_signal, filtered_signal, i, ToA)
%     grid on;
%     ylim([-1.5,1.5])
% end
% nexttile;


for i = 1:num_sensors
    % Extract raw signal
    raw_signal = sensor_data(:, i);

    % Apply bandpass filter
    filtered_signal = lowpass_filter(raw_signal, time);
    %ToA = compute_ToA(filtered_signal, time, 'sil');

    % Plot raw and filtered signal
    plot(time,raw_signal)
    hold on
    grid on;
    ylim([-1.5,1.5])
end
xlabel('Time (s)');
ylabel('Voltage (V)');
title('Signal from all sensors');
grid on;
hold off
ylim([-1,1])
% force_filter = lowpass_filter(force_data,time);
% plot(time, force_data, 'r', 'LineWidth', 1, 'DisplayName', 'Raw Signal');
% hold on
% plot(time, force_filter, 'b', 'LineWidth', 1, 'DisplayName', 'Filtered Signal');
% xlabel('Time (s)');
% ylabel('Voltage (V)');
% title('Force');
% legend;
% grid on;
% hold off







%%
%%% ==========================
%%%   FUNCTION TO READ FORCE DATA
%%% ==========================
function force_data = read_force_data(force_filename)
    % Read the force data file
    fid = fopen(force_filename, 'r');
    if fid == -1
        error('Cannot open force data file: %s', force_filename);
    end

    % Skip header lines until the second occurrence of '***End_of_Header***'
    header_count = 0;
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, '***End_of_Header***')
            header_count = header_count + 1;
        end
        if header_count == 2
            break;
        end
    end

    % Read column names to detect the number of columns
    column_names = strsplit(fgetl(fid), '\t');
    num_columns = length(column_names)-1;

    % Generate formatSpec dynamically
    formatSpec = repmat('%f ', 1, num_columns);
    formatSpec = strtrim(formatSpec);

    % Read the force data
    data = textscan(fid, formatSpec, 'Delimiter', '\t');
    fclose(fid);

    % Assume the force data is in the last column
    force_data = cell2mat(data(end)); 
end
