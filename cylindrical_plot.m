clear, clc, close all
% Load data
data = load("\\wsl.localhost\Ubuntu\home\tharm\FYP_Python\predictions_XGB.mat");

% Extract variables
y_test = data.y_test;
y_pred_x = data.y_pred_x;
y_pred_y = data.y_pred_y;
rmse_x = data.rmse(1);
rmse_y = data.rmse(2);
rmse_total = data.rmse(3);



%% Cylindrical------------------------------------------
figure;
% Parameters


r = 11.55;  % Radius
z_max = 45; % Height

% Cylinder generation
theta = linspace(0, 2*pi, 100);
z = linspace(0, z_max, 100);
[Theta, Z] = meshgrid(theta, z);
X = r * cos(Theta);
Y = r * sin(Theta);


% Plot ML Prediction

scatter3(r*cos(y_test(:,1)), r*sin(y_test(:,1)),y_test(:,2), 100, 'b+', 'LineWidth', 2); % True positions as blue +
hold on
scatter3(r*cos(y_pred_x),r*sin(y_pred_x), y_pred_y, 100, 'r+', 'LineWidth', 2); % Predicted positions as red +

% Draw Grey Dashed Lines Connecting True and Predicted Points
for i = 1:length(y_test)
    % Get true and predicted positions in Cartesian coordinates
    true_x = r * cos(y_test(i,1));
    true_y = r * sin(y_test(i,1));
    true_z = y_test(i,2);

    pred_x = r * cos(y_pred_x(i));
    pred_y = r * sin(y_pred_x(i));
    pred_z = y_pred_y(i);

    % Plot dashed grey line connecting each true-predicted pair
    plot3([true_x, pred_x], [true_y, pred_y], [true_z, pred_z], 'k--', 'LineWidth', 1.2); 
end

%----------------Cylindrical----------
surf(X, Y, Z, 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'FaceColor', [0.5 0.5 0.5]); % Solid gray with transparency


% **Grid Lines: Divide the Cylinder**
% Divide Z direction into 5 sections
z_divisions = linspace(0, z_max, 7); % 6 sections → 7 division points
theta_grid = linspace(0, 2*pi, 100); % Fine spacing for smooth curves

for i = 1:length(z_divisions)
    X_grid = r * cos(theta_grid);
    Y_grid = r * sin(theta_grid);
    Z_grid = ones(size(theta_grid)) * z_divisions(i);
    plot3(X_grid, Y_grid, Z_grid, 'k-', 'LineWidth', 0.3); % Black horizontal grid lines
end

% Divide Circumference into 8 sections
theta_divisions = linspace(0, 2*pi, 9); % 8 sections → 9 division points
z_grid = linspace(0, z_max, 100); % Full height

for i = 1:length(theta_divisions)
    X_grid = r * cos(theta_divisions(i)) * ones(size(z_grid));
    Y_grid = r * sin(theta_divisions(i)) * ones(size(z_grid));
    Z_grid = z_grid;
    plot3(X_grid, Y_grid, Z_grid, 'k-', 'LineWidth', 0.3); % Black vertical grid lines
end

% Blue square markers at z = 0 and z = 45, theta = 0, 90, 180, 270
theta_sensor = deg2rad([0, 90, 180, 270]);
z_sensor = [0, z_max];
for z_val = z_sensor
    for theta_val = theta_sensor
        x_sensor = r * cos(theta_val);
        y_sensor = r * sin(theta_val);
        plot3(x_sensor, y_sensor, z_val, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'black');
    end
end



% Annotate RMSE on the Plot
% Main title (bold)
title_text = 'True vs Predicted Positions';

% RMSE text (smaller and not bold)
rmse_text = sprintf('\\fontsize{10}RMSE: %.3f mm (X: %.3f, Y: %.3f)', ...
    rmse_total, rmse_x, rmse_y);

% Combine and set title with formatting
title({['\bf ' title_text], rmse_text}, 'FontSize', 14);


legend('True Position', 'Predicted Position','Location','best');

% Axis settings
axis equal;
xlabel('X-axis (mm)');
ylabel('Y-axis (mm)');
zlabel('Z-axis (mm)');
grid on;
hold off;



%% Map 2D 
%% Map 2D 

% Define cylinder properties
r = 11.55;            % Cylinder radius
height = 45;          % Cylinder height (Z range)
circumference = 2 * pi * r; % Unwrapped width

% Convert data: X = Z, Y = Unwrapped Theta
true_x_flat = y_test(:,2);  % Z-axis remains unchanged
true_y_flat = r * y_test(:,1);  % Unwrapped theta coordinate

pred_x_flat = y_pred_y;  % Predicted Z values remain unchanged
pred_y_flat = r * y_pred_x;  % Unwrapped theta coordinate for predictions

% Plot true and predicted positions
scatter(true_x_flat, true_y_flat, 100, 'b+', 'LineWidth', 2); % True positions as blue +
hold on;
scatter(pred_x_flat, pred_y_flat, 100, 'r+', 'LineWidth', 2); % Predicted positions as red +

% Draw Grey Dashed Lines Connecting True and Predicted Points
for i = 1:length(y_test)
    plot([true_x_flat(i), pred_x_flat(i)], [true_y_flat(i), pred_y_flat(i)], ...
          'k--', 'LineWidth', 1.2); % Dashed grey line
end

% **Directly Define the Rectangle Frame (Centered)**
x = [0, height, height, 0, 0]; % Z range (X-axis)
y = [circumference/2, circumference/2, -circumference/2, -circumference/2, circumference/2]; % Centered Y

% Plot the unwrapped rectangle
plot(x, y, 'k-', 'LineWidth', 2); % Black rectangle outline

% Define grid lines
num_h_lines = 7; % Number of horizontal grid lines
num_v_lines = 5; % Number of vertical grid lines

% **Horizontal grid lines (Centered)**
for i = 1:num_h_lines
    y_h = circumference/2 - i * (circumference / (num_h_lines + 1));
    plot([0, height], [y_h, y_h], 'k-', 'LineWidth', 0.5); % Black horizontal grid lines
end

% **Vertical grid lines (Z divisions)**
for j = 1:num_v_lines
    x_v = j * (height / (num_v_lines + 1));
    plot([x_v, x_v], [-circumference/2, circumference/2], 'k-', 'LineWidth', 0.5); % Black vertical grid lines
end

% **Sensor Positions (Centered)**
sensor_theta = linspace(-pi, pi, 5); % 8 segments around circumference
sensor_positions = [
    zeros(5,1), -r * sensor_theta(1:5)';  % Left edge sensors (S1-S5)
    height * ones(5,1), -r * sensor_theta(1:5)';  % Right edge sensors (S6-S10)
];

% Define sensor labels
S_labels = {'S1', 'S2', 'S3', 'S4','S1', 'S5', 'S6', 'S7', 'S8', 'S5'};

% Add blue text labels for sensor positions
for k = 1:length(S_labels)
    text(sensor_positions(k,1), sensor_positions(k,2), S_labels{k}, 'Color', 'b', ...
        'FontSize', 15, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

% Annotate RMSE on the Plot
title_text = 'Flattened Cylinder: True vs Predicted Positions';
rmse_text = sprintf('\\fontsize{10}RMSE: %.3f mm (X: %.3f, Y: %.3f)', ...
    rmse_total, rmse_x, rmse_y);

title({['\bf ' title_text], rmse_text}, 'FontSize', 14);

legend('True Position', 'Predicted Position', 'Location', 'northeast');

% Axis settings
xlabel('Z-axis (Height) (mm)');
ylabel('Unwrapped Circumference Position (mm)');
xlim([0-10, height+10]);
ylim([-circumference/2-10, circumference/2+10]); % Centered frame
grid on;
hold off;
