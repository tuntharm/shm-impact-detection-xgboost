%% Verify Velocity
%% 1 ToA
clear, clc
filename = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\tank\silham_p1_tank_new\silham_l6_p1_tank_1.txt"
[time, sensor_data,force_data] = read_labview_file(filename);
impact_type_match = regexp(filename, '(stl|sil|rub|srub)', 'match');
impact_type = impact_type_match{1};
probe_match = regexp(filename, 'p(\d+)', 'tokens'); % Find "p" followed by a number
probe = str2double(probe_match{1}{1}); % Convert to number

raw_signal = sensor_data(:,1);
num_sensors = size(sensor_data, 2);
for i = 1:num_sensors
    % Extract raw signal
    raw_signal = sensor_data(:, i);
    
    % Apply bandpass filter
    filtered_signal = lowpass_filter(raw_signal, time);
    ToA(i) = compute_ToA(filtered_signal, time, impact_type);
end
ToA = ToA-min(ToA);

%% 2 Distance
r = 11.55;
width = 45;             % Horizontal length
height = 2 * pi * r; % Vertical length

S_positions = [
    0, 0;                              % S1
    0, -2 * (2 * pi * r / 8);          % S2
    0, -4 * (2 * pi * r / 8);          % S3
    0, -6 * (2 * pi * r / 8);          % S4
    -width, 0;                         % S5
    -width, -2 * (2 * pi * r / 8);     % S6
    -width, -4 * (2 * pi * r / 8);     % S7
    -width, -6 * (2 * pi * r / 8);      % S8
];

for i = 1:num_sensors
    S(i) = sqrt((-7.5-S_positions(i,1))^2+S_positions(i,2)^2);
    V(i) = S(i)/ToA(i);
end
