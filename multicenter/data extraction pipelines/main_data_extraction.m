folder_Path ='PATH_TO_DATA\138_HealthyPiG'; 
%input(prompt);
%cd(folder_Path);
extracted_data = [];

subject_Folder = dir(folder_Path); %Path to all the Patients A, B-C, E, ....
for i=3:size(subject_Folder,1)


    pat_Folder = dir([folder_Path, '\',  subject_Folder(i).name]);
    c3d_Files = dir([folder_Path, '\',  subject_Folder(i).name, '\', '/*.c3d']);

    for c3d_file=2:size(c3d_Files, 1)
        data = get_intellevent_data_grf(c3d_Files(c3d_file));

        trial_name = split(c3d_Files(c3d_file).name, ' ');
        data.Trial = c3d_Files(c3d_file).name;
        data.DBid = trial_name(1);
        data.Label = 1;
        % age, weight, height, OFM, sex, condition
        if size(data.Trajectory, 2) > 100
            extracted_data = [extracted_data; data];
        end
    end


end
%%
folder_Path ='PATH_TO_DATA\MCA\50_StrokePiG';%'C:\FHSTP\O3DGA_Projects\Data Extraction\MCA\50_StrokePiG';

subject_Folder = dir(folder_Path); %Path to all the Patients A, B-C, E, ....
for i=4:size(subject_Folder,1)


    pat_Folder = dir([folder_Path, '\',  subject_Folder(i).name]);
    c3d_Files = dir([folder_Path, '\',  subject_Folder(i).name, '\', '/*.c3d']);

    for c3d_file=1:size(c3d_Files, 1)-1
        data = get_intellevent_data_grf(c3d_Files(c3d_file));

        trial_name = split(c3d_Files(c3d_file).name, ' ');
        data.Trial = c3d_Files(c3d_file).name;
        data.DBid = subject_Folder(i).name;
        data.Label = 2;
        % age, weight, height, OFM, sex, condition
        
        if size(data.Trajectory, 2) > 100
            extracted_data = [extracted_data; data];
        end
    end


end

%% ONLY TRIALS WHERE GRF is AVAILABLE

keepIdx = false(length(extracted_data), 1);

for i = 1:length(extracted_data)
    % Check if not all elements are zero
    if any(extracted_data(i).GRF_Events ~= 0)
        keepIdx(i) = true;
    end
end

% Keep only entries with non-zero elements
extracted_data = extracted_data(keepIdx);