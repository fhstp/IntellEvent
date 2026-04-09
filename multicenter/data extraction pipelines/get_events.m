function events = get_events(c3d, varargin)
    % Returns events as indices of frames or as time in seconds
    % Inputs: - type: string of either 'idx' or 'time'. With 'idx'
    %                 the output is in indices. With 'time' the
    %                 output is in seconds.
    %         - frequency: String of either 'point' or 'analog'.
    %                      Default is 'point'. Specifies if events should be
    %                      output in the frequency of btkPoint (normally 200Hz)
    %                      or of analog channels (normally 2000Hz)
    % Output 'events' is a structure with fields:
    %   - left_fs: Left foot strikes
    %   - left_fo: Left foot off
    %   - right_fs: Right foot strike
    %   - right_fo: Right foot off
    %   - general: General events
    
    % Parse inputs
    p = inputParser;
    addRequired(p, 'c3d')
    addParameter(p, 'type', 'idx', @(x) any(validatestring(x,{'idx', 'time'})))
    addParameter(p, 'frequency', 'point', @(x) any(validatestring(x,{'point', 'analog'})))
    parse(p,c3d,varargin{:})
    
    % Get sampling frequencies, first frame and events
    acq = btkReadAcquisition(c3d);
    freq_analog = btkGetAnalogFrequency(acq);
    freq_point = btkGetPointFrequency(acq);
    analog_samples_per_frame = btkGetAnalogSampleNumberPerFrame(acq);
    first_frame_analog = btkGetFirstFrame(acq) * analog_samples_per_frame;
    ev = btkGetEvents(acq); % Events from btk are returned in seconds and the first frame is 0.0s
    btkDeleteAcquisition(acq);

    % Go through events
    for ev_name = {'Left_Foot_Strike', 'Left_Foot_Off', 'Right_Foot_Strike',...
                                      'Right_Foot_Off', 'General_Event'}
        % Create empty array for events that do not exist
        if ~isfield(ev, ev_name{1})
            ev.(ev_name{1}) = [];
            continue
        end
        if strcmp(p.Results.type, 'idx')
            if strcmp(p.Results.frequency, 'point')
                % It is +2 because btk outputs 0.0s for the first
                % frame. If the time for the first frame would be
                % 1/sampling_freq, then it would be correct with +1
                ev.(ev_name{1}) = round((ev.(ev_name{1}) * freq_analog - first_frame_analog)/analog_samples_per_frame) + 2;   
            elseif strcmp(p.Results.frequency, 'analog')
                % Indices in analog frequency
                ev.(ev_name{1}) = round(ev.(ev_name{1}) * freq_analog- (first_frame_analog-analog_samples_per_frame)) + 1;
            end
        elseif strcmp(p.Results.type, 'time')
            % Time in analog frequency accuracy
            ev.(ev_name{1}) = ev.(ev_name{1}) - ((first_frame_analog-analog_samples_per_frame)/freq_analog);
            if strcmp(p.Results.frequency, 'point')
                % Downsample time to point frequency
                ev.(ev_name{1}) = round(ev.(ev_name{1}) * freq_analog / analog_samples_per_frame) / freq_point;
            end
        end
    end

    % Assign events to structure with easier fieldnames
    events.left_fs = ev.Left_Foot_Strike;
    events.left_fo = ev.Left_Foot_Off;
    events.right_fs = ev.Right_Foot_Strike;
    events.right_fo = ev.Right_Foot_Off;
    events.general = ev.General_Event;
end