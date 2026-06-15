%% Verify Velocity
%% 1 ToA
clear, clc
filename = "data/raw/tank/stlham_p1_tank_new/stlham_l1_p1_tank_1.txt"
[time, sensor_data,force_data] = read_labview_file(filename);
impact_type_match = regexp(filename, '(stl|sil|rub|srub)', 'match');
impact_type = impact_type_match{1};
probe_match = regexp(filename, 'p(\d+)', 'tokens'); % Find "p" followed by a number
probe = str2double(probe_match{1}{1}); % Convert to number
loc_match = regexp(filename, 'l(\d+)', 'tokens');
loc = str2double(loc_match{1}{1}); % Convert to number

%% 2 Plot Figure for raw vs. filtered signals
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



%% Test 

num_sensors = size(sensor_data, 2);
for i = 1:num_sensors
    % Extract raw signal
    raw_signal = sensor_data(:, i);
    
    % Apply bandpass filter
    filtered_signal = lowpass_filter(raw_signal, time);
    ToA(i) = compute_ToA(filtered_signal, time, impact_type);
end
dToA = ToA-min(ToA); % Different Time of arrival
% 2 Distance
r = 11.55;
width = 45;             % Horizontal length
height = 2 * pi * r; % Vertical length



S_positions_cir = [
0,0; 
0, -2 * (2 * pi * r / 8);
0, -4 * (2 * pi * r / 8);
0, 2 * (2 * pi * r / 8);
-width, 0;    
-width, -2 * (2 * pi * r / 8);     % S6
-width, -4 * (2 * pi * r / 8);     % S7
-width, 2 * (2 * pi * r / 8);      % S8
];
for i = 1:num_sensors
    dS(i) = sqrt((-7.5-S_positions_cir(i,1))^2+S_positions_cir(i,2)^2)-7.5;
    V(i) = dS(i)/dToA(i);
end

for i = 1:num_sensors;
    angle(i) = atan(S_positions_cir(i,2)/(7.5+S_positions_cir(i,1)));

    % Compute angle based on cylindrical unwrapping
    if i == 1
        angle(i) = 0; % Sensor 1 reference at 0 degrees
    elseif i == 5
        angle(i) = 180; % Sensor 5 should be at 180 degrees
    else
        angle(i) = atan2d(S_positions_cir(i,2), (7.5 + S_positions_cir(i,1))); 
        % Convert to positive angles
        if angle(i) < 0
            angle(i) = angle(i) + 360;
        end
    end
end

% Polar plot of velocity vs. angle
figure;
polarplot(deg2rad(angle), V, 'ko', 'LineWidth', 1.5, 'MarkerFaceColor', 'w'); % Experiment (circle markers)
hold on 
polarplot(2*pi-deg2rad(angle(3)), V(3), 'ko', 'LineWidth', 1.5, 'MarkerFaceColor', 'w'); % Experiment (circle markers)
polarplot(2*pi-deg2rad(angle(7)), V(7), 'ko', 'LineWidth', 1.5, 'MarkerFaceColor', 'w'); % Experiment (circle markers)


% Formatting
legend('Experiment', 'Location', 'best');
title('Polar Plot of Experimental Data (Cylindrical Correction)');