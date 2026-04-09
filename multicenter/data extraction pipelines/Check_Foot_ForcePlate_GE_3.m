function [LFS_f,LFO_f,RFS_f,RFO_f] = Check_Foot_ForcePlate_GE_2(fp_corners,m,foot_wide_threshold,foot_front_threshold,FP_FS_f,FP_FO_f,ff,idx_x,idx_y)
% description: Detection of foot steps clean inside force platforms (FP)
%
% input:
% fp_corners - spatial coordinates of FP corners (matrix 3x4)
% m - structure m containing marker coordinates
% foot_wide_threshold - increment to foot wide
% foot_front_threshold - increment to foot length
% FP_FS_f - foot strike frame
% FP_FO_f - foot off frame
% ff - first frame of acquisition
%
% output:
% LFS_f/RFS_f - validity of step for the foot strike event (L-left/R-right)
% LFO_f/RFO_f - validity of step for the foot off event (L-left/R-right)

% Modifications (by Marys Revaz & Xavier Gasparutto on July 2025) : 
% - Modify computation of rectangle around foot (in direction of heel-toe vector)
% - Modify computation of foot length (in direction of heel-toe vector)
% - Modify way of checking if foot is inside platform (checking entire
% rectangle instead of only 4 corners)

%% FOOT STRIKE
% Get the coordinates of foot markers on the frame of interest
RHEE_FS = m.RHEE(FP_FS_f-ff,:); 
LHEE_FS = m.LHEE(FP_FS_f-ff,:);
RTOE_FS = m.RTOE(FP_FS_f-ff,:); 
LTOE_FS = m.LTOE(FP_FS_f-ff,:); 

% initialize output variables
LFS_f = [];
RFS_f = [];
LFO_f = [];
RFO_f = [];

% size of the feet at foot strike
Length_LFoot = pdist2(LHEE_FS(1, [idx_x, idx_y]), LTOE_FS(1, [idx_x, idx_y]), 'euclidean')*foot_front_threshold;
Length_RFoot = pdist2(RHEE_FS(1, [idx_x, idx_y]), RTOE_FS(1, [idx_x, idx_y]), 'euclidean')*foot_front_threshold;
Left_width = (foot_wide_threshold*Length_LFoot)/4;
Right_width = (foot_wide_threshold*Length_RFoot)/4;

% define foot rectangles
L_v_heel_toe = (LTOE_FS(1, [idx_x, idx_y]) - LHEE_FS(1, [idx_x, idx_y]));
L_v_norm_heel_toe = L_v_heel_toe/norm(L_v_heel_toe);
L_v_mediolat = cross([0, 0, 1], [L_v_norm_heel_toe, 0]);
L_v_mediolat = L_v_mediolat(1, 1:2);
L_v_norm_mediolat = L_v_mediolat/norm(L_v_mediolat);
lf_corners = [LHEE_FS(1, [idx_x, idx_y]) + L_v_norm_mediolat*Left_width; ...
    LHEE_FS(1, [idx_x, idx_y]) + L_v_norm_mediolat*Left_width + L_v_norm_heel_toe*Length_LFoot; ...
    LHEE_FS(1, [idx_x, idx_y]) - L_v_norm_mediolat*Left_width + L_v_norm_heel_toe*Length_LFoot; ...
    LHEE_FS(1, [idx_x, idx_y]) - L_v_norm_mediolat*Left_width];

R_v_heel_toe = (RTOE_FS(1, [idx_x, idx_y]) - RHEE_FS(1, [idx_x, idx_y]));
R_v_norm_heel_toe = R_v_heel_toe/norm(R_v_heel_toe);
R_v_mediolat = cross([0, 0, 1], [R_v_norm_heel_toe, 0]);
R_v_mediolat = R_v_mediolat(1, 1:2);
R_v_norm_mediolat = R_v_mediolat/norm(R_v_mediolat);
rf_corners = [RHEE_FS(1, [idx_x, idx_y]) + R_v_norm_mediolat*Right_width; ...
    RHEE_FS(1, [idx_x, idx_y]) + R_v_norm_mediolat*Right_width + R_v_norm_heel_toe*Length_RFoot; ...
    RHEE_FS(1, [idx_x, idx_y]) - R_v_norm_mediolat*Right_width + R_v_norm_heel_toe*Length_RFoot; ...
    RHEE_FS(1, [idx_x, idx_y]) - R_v_norm_mediolat*Right_width];

lf_rectangle = polyshape([lf_corners(1,1), lf_corners(2,1), lf_corners(3,1), lf_corners(4,1), lf_corners(1,1)], ...
    [lf_corners(1,2), lf_corners(2,2), lf_corners(3,2), lf_corners(4,2), lf_corners(1,2)]);
rf_rectangle = polyshape([rf_corners(1,1), rf_corners(2,1), rf_corners(3,1), rf_corners(4,1), rf_corners(1,1)], ...
    [rf_corners(1,2), rf_corners(2,2), rf_corners(3,2), rf_corners(4,2), rf_corners(1,2)]);
fp_rectangle = polyshape([fp_corners(idx_x,1), fp_corners(idx_x,2), fp_corners(idx_x,3), fp_corners(idx_x,4), fp_corners(idx_x,1)], ...
    [fp_corners(idx_y,1), fp_corners(idx_y,2), fp_corners(idx_y,3), fp_corners(idx_y,4), fp_corners(idx_y,1)]);

% check if foot rectangles are fully inside and fully outside FP rectangle
L_FS_inside = isequal(area(lf_rectangle), area(intersect(lf_rectangle, fp_rectangle)));
L_FS_outside = (area(intersect(lf_rectangle, fp_rectangle)) == 0);
R_FS_inside = isequal(area(rf_rectangle), area(intersect(rf_rectangle, fp_rectangle)));
R_FS_outside = (area(intersect(rf_rectangle, fp_rectangle)) == 0);

if R_FS_inside && L_FS_outside % if R foot fully inside and L foot fully outside
    RFS_f = [RFS_f, FP_FS_f];
elseif L_FS_inside && R_FS_outside % if L foot fully inside and R foot fully outside
    LFS_f = [LFS_f, FP_FS_f];
end
    
%% FOOT OFF (same principle of Foot strike)
% Get the coordinates of foot markers on the frame of interest
RHEE_FO = m.RHEE(FP_FO_f-ff,:);                        
LHEE_FO = m.LHEE(FP_FO_f-ff,:);
RTOE_FO = m.RTOE(FP_FO_f-ff,:); 
LTOE_FO = m.LTOE(FP_FO_f-ff,:);

% size of the feet at foot off
Length_LFoot = pdist2(LHEE_FO(1, [idx_x, idx_y]), LTOE_FO(1, [idx_x, idx_y]), 'euclidean')*foot_front_threshold;
Length_RFoot = pdist2(RHEE_FO(1, [idx_x, idx_y]), RTOE_FO(1, [idx_x, idx_y]), 'euclidean')*foot_front_threshold;

% define foot rectangles
L_v_heel_toe = (LTOE_FO(1, [idx_x, idx_y]) - LHEE_FO(1, [idx_x, idx_y]));
L_v_norm_heel_toe = L_v_heel_toe/norm(L_v_heel_toe);
L_v_mediolat = cross([0, 0, 1], [L_v_norm_heel_toe, 0]);
L_v_mediolat = L_v_mediolat(1, 1:2);
L_v_norm_mediolat = L_v_mediolat/norm(L_v_mediolat);
lf_corners = [LHEE_FO(1, [idx_x, idx_y]) + L_v_norm_mediolat*Left_width; ...
    LHEE_FO(1, [idx_x, idx_y]) + L_v_norm_mediolat*Left_width + L_v_norm_heel_toe*Length_LFoot; ...
    LHEE_FO(1, [idx_x, idx_y]) - L_v_norm_mediolat*Left_width + L_v_norm_heel_toe*Length_LFoot; ...
    LHEE_FO(1, [idx_x, idx_y]) - L_v_norm_mediolat*Left_width];

R_v_heel_toe = (RTOE_FO(1, [idx_x, idx_y]) - RHEE_FO(1, [idx_x, idx_y]));
R_v_norm_heel_toe = R_v_heel_toe/norm(R_v_heel_toe);
R_v_mediolat = cross([0, 0, 1], [R_v_norm_heel_toe, 0]);
R_v_mediolat = R_v_mediolat(1, 1:2);
R_v_norm_mediolat = R_v_mediolat/norm(R_v_mediolat);
rf_corners = [RHEE_FO(1, [idx_x, idx_y]) + R_v_norm_mediolat*Right_width; ...
    RHEE_FO(1, [idx_x, idx_y]) + R_v_norm_mediolat*Right_width + R_v_norm_heel_toe*Length_RFoot; ...
    RHEE_FO(1, [idx_x, idx_y]) - R_v_norm_mediolat*Right_width + R_v_norm_heel_toe*Length_RFoot; ...
    RHEE_FO(1, [idx_x, idx_y]) - R_v_norm_mediolat*Right_width];

lf_rectangle = polyshape([lf_corners(1,1), lf_corners(2,1), lf_corners(3,1), lf_corners(4,1), lf_corners(1,1)], ...
    [lf_corners(1,2), lf_corners(2,2), lf_corners(3,2), lf_corners(4,2), lf_corners(1,2)]);
rf_rectangle = polyshape([rf_corners(1,1), rf_corners(2,1), rf_corners(3,1), rf_corners(4,1), rf_corners(1,1)], ...
    [rf_corners(1,2), rf_corners(2,2), rf_corners(3,2), rf_corners(4,2), rf_corners(1,2)]);
fp_rectangle = polyshape([fp_corners(idx_x,1), fp_corners(idx_x,2), fp_corners(idx_x,3), fp_corners(idx_x,4), fp_corners(idx_x,1)], ...
    [fp_corners(idx_y,1), fp_corners(idx_y,2), fp_corners(idx_y,3), fp_corners(idx_y,4), fp_corners(idx_y,1)]);

% check if foot rectangles are fully inside and fully outside FP rectangle
L_FO_inside = isequal(area(lf_rectangle), area(intersect(lf_rectangle, fp_rectangle)));
L_FO_outside = (area(intersect(lf_rectangle, fp_rectangle)) == 0);
R_FO_inside = isequal(area(rf_rectangle), area(intersect(rf_rectangle, fp_rectangle)));
R_FO_outside = (area(intersect(rf_rectangle, fp_rectangle)) == 0);

if R_FO_inside && L_FO_outside % if R foot fully inside and L foot fully outside
    RFO_f = [RFO_f, FP_FO_f];
elseif L_FO_inside && R_FO_outside % if L foot fully inside and R foot fully outside
    LFO_f = [LFO_f, FP_FO_f];
end

end
