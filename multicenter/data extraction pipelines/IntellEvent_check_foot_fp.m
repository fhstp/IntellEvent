function [Lfoot, Rfoot] = IntellEvent_check_foot_fp(acq, metadata, rate_point, rate_analog, firstFrame_point, lastFrame_point, idx_x, idx_y)
% Compute GRF events (1 foot completely inside platform)
% (developed by Marys Franco Carvalho on January 2024 based on Mickael Check_Foot_ForcePlate_3 code)

% adding code folders to the MATLAB path
% GaitEvent_DataSelection toolbox
addpath(genpath('C:\Users\francoca\OneDrive - unige.ch\Documents\MATLAB\GEV\Scripts'))

%% 1.0 Define inital threshold parameters
% threshold for GRF detection of FS and FO
threshold_FP_detection = 20;
% foot_wide_threshold for clean step detection, default 1.2 (increment of 20% of foot width)
foot_wide_threshold = 1.2;
% foot_front_threshold for clean step detection, default 1.1 (increment of 10% of foot length [heel-toe + 10%])
foot_front_threshold = 1.1;

Lfoot = [];
Rfoot = [];
n = 1;

%% 1.1 Get data
m = btkGetMarkers(acq);
fp = btkGetForcePlatforms(acq);
% Count the number of force plates
nPF_used = metadata.children.FORCE_PLATFORM.children.USED.info.values;

% create SACR marker if missing (midpoint between 2 PSI)
if ~isfield(m, 'SACR') && isfield(m, 'LPSI') && isfield(m, 'RPSI')
    m.SACR = (m.LPSI' + m.RPSI').'/2;
end

%% 1.2 Get sense (sense=1 -> pf 1 to 2, sense=-1 -> pf 2 to 1)
% Sense/direction of trial
idx_notnan = find(~isnan(m.SACR(:, idx_x)));
dist_x = m.SACR(idx_notnan(end), idx_x) - m.SACR(idx_notnan(1), idx_x);
if dist_x >= 0
    sense = 1; % from plate 1 to 2
else
    sense = -1; % from plate 2 to 1
end

%% 2 Detection
for nPF = 1:nPF_used
    disp(['     - Platform: ', num2str(nPF)])
    % Force data (only vertical component) from ForcePlatforms 
    if isfield(fp(nPF).channels, 'Fz')
        FP = fp(nPF).channels.('Fz');
    elseif isfield(fp(nPF).channels, string(strcat('Fz',num2str(nPF))))
        FP = fp(nPF).channels.(strcat('Fz',num2str(nPF)));
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
            FP_FS_f = round((FP_detect(1).*rate_point)./rate_analog)+firstFrame_point;
            FP_FO_f = round((FP_detect(end).*rate_point)./rate_analog)+firstFrame_point;
            if FP_FS_f > firstFrame_point && FP_FO_f < lastFrame_point
                % get 4 corners of platform
                Corners_FP = fp(nPF).corners;
                % check if foot valid in platform(nFP)
                [lfs_f, lfo_f, rfs_f, rfo_f] = Check_Foot_ForcePlate_GE_2(Corners_FP,m,sense,foot_wide_threshold,foot_front_threshold,FP_FS_f,FP_FO_f,firstFrame_point,idx_x,idx_y);
                
                %% Left side
                if ~isempty(lfs_f) && ~isempty(lfo_f)
                    Lfoot(n).FP_number = nPF;
                    Lfoot(n).LFS = lfs_f - firstFrame_point + 1;
                    Lfoot(n).LFO = lfo_f - firstFrame_point + 1;
                end
                
                %% Right side
                if ~isempty(rfs_f) && ~isempty(rfo_f)
                    Rfoot(n).FP_number = nPF;
                    Rfoot(n).RFS = rfs_f - firstFrame_point + 1;
                    Rfoot(n).RFO = rfo_f - firstFrame_point + 1;
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

end