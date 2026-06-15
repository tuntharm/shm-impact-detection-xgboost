function [time, sensor_data, force_data] = read_labview_file_plate(filename)
    % Function to read LabVIEW measurement file with or without a header
    % Input: filename - Name of the main sensor data file
    % Output: time, sensor_data, force_data

    % Open file to check for '***End_of_Header***'
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    header_count = 0;
    line_number = 0;
    has_header = false;

    while ~feof(fid)
        line = fgetl(fid);
        line_number = line_number + 1;
        if contains(line, '***End_of_Header***')
            header_count = header_count + 1;
            has_header = true;
        end
        if header_count == 2
            break;
        end
    end
    fclose(fid);

    % Read data using the correct method
    if has_header
        T = readtable(filename, 'Delimiter', '\t', 'HeaderLines', line_number);
        % Extract time and sensor readings
        time = T{:, 1} - min(T{:, 1}); % First column is time
        sensor_data = T{:, 2:end-2};   % Extract all sensor data (excluding time and last column)
        force_data = T{:, end-1};        % Extract last column (force data)
    else
        T = readtable(filename); % Read normally if no header
            % Extract time and sensor readings
        time = T{:, 1} - min(T{:, 1}); % First column is time
        sensor_data = T{:, 2:end-1};   % Extract all sensor data (excluding time and last column)
        force_data = T{:, end};        % Extract last column (force data)

    end


end
