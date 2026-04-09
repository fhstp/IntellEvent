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



%% 1.2 Get sense (sense=1 -> pf 1 to 2, sense=-1 -> pf 2 to 1)
% CHANGE ACCORDINGLY TO THE LABORATORY COORDINATE SYSTEM!
idx_x = 1;
idx_y = 2;
% create SACR marker if missing (midpoint between 2 PSI)
m = btkGetMarkers(acq);

oldStruct = m;
newStruct = struct();

fields = fieldnames(oldStruct);

for i = 1:numel(fields)
    oldName = fields{i};
    
    if contains(oldName, '_')  % Only rename if an underscore is present
        parts = split(oldName, '_');
        newName = parts{end};  % Take the part after the last underscore
    else
        newName = oldName;  % Keep the name unchanged
    end
    
    % If newName already exists, issue a warning or handle conflicts
    if isfield(newStruct, newName)
        warning('Duplicate field name after renaming: %s. Skipping.', newName);
    else
        try
            newStruct.(newName) = oldStruct.(oldName);
        catch
            disp(oldName)
        end
    end
end

% Replace original struct if desired
m = newStruct;

if ~isfield(m, 'SACR') && isfield(m, 'LPSI') && isfield(m, 'RPSI')
    m.SACR = (m.LPSI' + m.RPSI').'/2;
end
% Sense/direction of trial
idx_notnan = find(~isnan(m.SACR(:, idx_x)));
dist_x = m.SACR(idx_notnan(end), idx_x) - m.SACR(idx_notnan(1), idx_x);
if dist_x >= 0
    sense = 1; % from plate 1 to 2
else
    sense = -1; % from plate 2 to 1
end


%% 2 Detection
n=1;
foot_wide_threshold = 1.1;
% foot_front_threshold for clean step detection, default 1.1 (increment of 10% of foot length [heel-toe + 10%])
foot_front_threshold = 1.3;

metadata = btkGetMetaData(acq);
fp = btkGetForcePlatforms(acq);
nPF_used = metadata.children.FORCE_PLATFORM.children.USED.info.values;
threshold_FP_detection = 20; % !Check with Gianna!
firstFrame_point = btkGetFirstFrame(acq);
lastFrame_point = btkGetLastFrame(acq);
freq_analog = btkGetAnalogFrequency(acq);
freq_point = btkGetPointFrequency(acq);
analog_samples_per_frame = btkGetAnalogSampleNumberPerFrame(acq);
first_frame_analog = btkGetFirstFrame(acq) * analog_samples_per_frame;

for nPF = 1:nPF_used
    errors = 0;
    disp(['     - Platform: ', num2str(nPF)])
    % Force data (only vertical component) from ForcePlatforms 
    if isfield(fp(nPF).channels, 'Fz')
        FP = fp(nPF).channels.('Fz');
    elseif isfield(fp(nPF).channels, string(strcat('Fz',num2str(nPF))))
        FP = fp(nPF).channels.(strcat('Fz',num2str(nPF)));
    elseif isfield(fp(nPF).channels, string(strcat('Force_Fz',num2str(nPF))))
        FP = fp(nPF).channels.(strcat('Force_Fz',num2str(nPF)));
    else
        continue
    end
    
   [~, idx_max] = max(abs(FP));
    if FP(idx_max) < 0
        FP = -FP;
    end
    
    % frame foot strike for each platform
    FP_detect = find(FP > threshold_FP_detection);
    if ~isempty(FP_detect)
        while (length(FP_detect) >= 10) % removing isolated cases with FP over the threshold due to noise
            if FP_detect(1)+9 == FP_detect(10)
                break
            else
                FP_detect = FP_detect(2:end);
            end
        end
        while (length(FP_detect) >= 10)
            if FP_detect(end-9)+9 == FP_detect(end)
                break
            else
                FP_detect = FP_detect(1:end-1);
            end
        end
        
        if ~isempty(FP_detect) && (length(FP_detect) >= 10)
            FP_FS_f = round((FP_detect(1).*freq_point)./freq_analog)+firstFrame_point;
            FP_FO_f = round((FP_detect(end).*freq_point)./freq_analog)+firstFrame_point;
            if FP_FS_f > firstFrame_point && FP_FO_f < lastFrame_point
                % get 4 corners of platform
                Corners_FP = fp(nPF).corners;
                % check if foot valid in platform(nFP)
                %[lfs_f, lfo_f, rfs_f, rfo_f] = Check_Foot_ForcePlate_GE_2(Corners_FP,m,sense,foot_wide_threshold,foot_front_threshold,FP_FS_f,FP_FO_f,firstFrame_point,idx_x,idx_y);
                 [lfs_f, lfo_f, rfs_f, rfo_f] = Check_Foot_ForcePlate_GE_3(Corners_FP,m,foot_wide_threshold,foot_front_threshold,FP_FS_f,FP_FO_f,firstFrame_point,idx_x,idx_y);
                %% Left side
                % try
                %     if string(fps(nPF)) == 'Left'
                %         Lfoot(n).FP_number = nPF;
                %         Lfoot(n).LFS = FP_FS_f - firstFrame_point + 1;
                %         Lfoot(n).LFO = FP_FO_f - firstFrame_point + 1;
                %         grf_events(FP_FS_f - firstFrame_point + 1) = 1;
                %         grf_events(FP_FO_f - firstFrame_point + 1) = 2;
                %     end
                % 
                %     %% Right side
                %     if string(fps(nPF)) == 'Right'
                %         Rfoot(n).FP_number = nPF;
                %         Rfoot(n).RFS = FP_FS_f - firstFrame_point + 1;
                %         Rfoot(n).RFO = FP_FO_f - firstFrame_point + 1;
                %         grf_events(FP_FS_f - firstFrame_point + 1) = 3;
                %         grf_events(FP_FO_f - firstFrame_point + 1) = 4;
                %     end
                % catch
                %     errors = 1;
                % end
                %% Left side
                if ~isempty(lfs_f) && ~isempty(lfo_f)
                   Lfoot(n).FP_number = nPF;
                   Lfoot(n).LFS = lfs_f - firstFrame_point + 1;
                   Lfoot(n).LFO = lfo_f - firstFrame_point + 1;
                   grf_events(lfs_f - firstFrame_point + 1) = 1;
                   grf_events(lfo_f - firstFrame_point + 1) = 2;
                end
                
                %% Right side
                if ~isempty(rfs_f) && ~isempty(rfo_f)
                   Rfoot(n).FP_number = nPF;
                   Rfoot(n).RFS = rfs_f - firstFrame_point + 1;
                   Rfoot(n).RFO = rfo_f - firstFrame_point + 1;
                   grf_events(rfs_f - firstFrame_point + 1) = 3;
                   grf_events(rfo_f - firstFrame_point + 1) = 4;
                end

            else
                disp('Warning: First frame equals to signal detected in platform!')
            end
        else
            disp('Warning: Platform could not be used! ')
        end
    else
        disp('Warning: Platform could not be used! ')
    end
    n = n+1;
end


% Close acquistion to free memory

btkDeleteAcquisition(acq)
data = struct('Trajectory', marker_data,...
              'All_Events', all_events,...
              'GRF_Events', grf_events,...
              'Freq_point', point_freq,...
              'Freq_analog', fp_freq);
end