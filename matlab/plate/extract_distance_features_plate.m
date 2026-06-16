function distance_features = extract_distance_features_plate(filename, sensorxy)
    % Extract impact location number from filename (assumes "l<number>" format)
    loc_str = regexp(filename, 'l(\d+)', 'tokens');
    if ~isempty(loc_str)
        loc = str2double(loc_str{1});
    else
        warning(['Unable to extract impact location from: ', filename]);
        distance_features = NaN;
        return;
    end

    % Map impact location number to (X, Y) coordinates
    [x, y] = map_location_to_xy(loc);
    % Combine into a row vector: [loc, x, y, r1, r2, ..., rN]
    distance_features = [loc, x, y];


    % Debugging Output
    %disp(['Debug: Extracted Distance Features for ', filename]);
    % disp(distance_features);
end

%%% ==========================
%%%  FUNCTION: Map Location to X, Y Coordinates
%%% ==========================
function [x, y] = map_location_to_xy(loc)
    if isnan(loc)
        x = NaN; y = NaN;
        return;
    end

    % Define X and Y mappings based on given logic
    if any(loc == [1, 8, 15, 22, 29]), x = -60;
    elseif any(loc == [2, 9, 16, 23, 30]), x = -40;
    elseif any(loc == [3, 10, 17, 24, 31]), x = -20;
    elseif any(loc == [4, 11, 18, 25, 32]), x = 0;
    elseif any(loc == [5, 12, 19, 26, 33]), x = 20;
    elseif any(loc == [6, 13, 20, 27, 34]), x = 40;
    elseif any(loc == [7, 14, 21, 28, 35]), x = 60;
    else, x = NaN;
    end

    if any(loc == [1, 2, 3, 4, 5, 6, 7]), y = -40;
    elseif any(loc == [8, 9, 10, 11, 12, 13, 14]), y = -20;
    elseif any(loc == [15, 16, 17, 18, 19, 20, 21]), y = 0;
    elseif any(loc == [22, 23, 24, 25, 26, 27, 28]), y = 20;
    elseif any(loc == [29, 30, 31, 32, 33, 34, 35]), y = 40;
    else, y = NaN;
    end
end
