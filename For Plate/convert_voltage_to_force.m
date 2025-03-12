function force_N = convert_voltage_to_force(force_data,impact_type)
% Converts voltage readings to force (N) using sensor sensitivity and gain
%
% Inputs:
%   force_data - Vector of voltage readings from force sensor

%
% Output:
%   force_N - Vector of force values in Newtons (N)

    % Convert voltage to force
    S = 2.75/1000; %Sensor sensitivity (V/N) 
    G = 1;
    switch lower(impact_type)
            case {'stl','sil'}
                probe = 40;
            case {'rub','srub'}
                probe = 100;
            otherwise
                error('Unknown impact type: %s', impact_type);
    end
    
    force_N = force_data ./ (G * S * probe);
    
end
