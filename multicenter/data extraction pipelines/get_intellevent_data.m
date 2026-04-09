function data = get_intellevent_data(c3d_file)
% Returns marker data, all events of a trial and force plate events of a
% trial

c3d = [c3d_file.folder, '\', c3d_file.name];
acq = btkReadAcquisition(c3d);

% Get marker data
markers = {'LHEE', 'LTOE', 'LANK', 'RHEE', 'RTOE', 'RANK'};
points = btkGetPoints(acq);
ofm = 0;

points_fields = fieldnames(points);

% Clean up the fieldnames: get only the marker part
clean_fields = regexprep(points_fields, '.*_(\w+):?$', '$1');

% Create a mapping: clean marker name → actual field name in `points`
marker_map = containers.Map();
for i = 1:length(clean_fields)
    marker_map(clean_fields{i}) = points_fields{i};
end

% Check if all markers exist
if ~all(ismember(markers, clean_fields))
    data = struct('Trajectory', [],...
              'All_Events', [],...
              'Freq_point', NaN,...
              'Freq_analog', NaN);
    return
end

% Extract marker data using the map
marker_data = [];
for i = 1:length(markers)
    true_field = marker_map(markers{i});  % Get actual field name
    marker_data = [marker_data; points.(true_field)'];  % Transpose and append
end

% Get frequencies
fp_freq = btkGetAnalogFrequency(acq);
point_freq = btkGetPointFrequency(acq);

% Get all available events
n = btkGetPointFrameNumber(acq);
events = get_events(c3d, 'type', 'idx');
all_events = zeros(1,n);
grf_events = zeros(1,n);
all_events(events.left_fs) = 1;
all_events(events.left_fo) = 2;
all_events(events.right_fs) = 3;
all_events(events.right_fo) = 4;

% Get forceplate data
% analogs = btkGetAnalogs(acq);


btkDeleteAcquisition(acq)
data = struct('Trajectory', marker_data,...
              'All_Events', all_events,...
              'Freq_point', point_freq,...
              'Freq_analog', fp_freq);
end