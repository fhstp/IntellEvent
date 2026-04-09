function inside = insidePlatform (corners, Platform)
% Check if a rectangle is inside another.
% Inside counts the number of points inside the platform. 
% (sum(inside) = 4 if all 4 points are inside. 
%
% input:
% corners - foot corners (3x4)
% Platform - FP corners (3x4)

% output:
% inside - check list (1x4)

min_x = min(Platform(1, :));
max_x = max(Platform(1, :));
min_y = min(Platform(2, :));
max_y = max(Platform(2, :));

% Check if each foot point lies within bounds
inside = corners(:,1) >= min_x & corners(:,1) <= max_x & ...
              corners(:,2) >= min_y & corners(:,2) <= max_y ;
end