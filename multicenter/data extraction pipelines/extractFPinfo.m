function fpInfo = extractFPinfo(filename)
    % Read the file content
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    % Initialize placeholders
    fpVals = cell(3,1);

    % Read file line by line
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        
        if startsWith(line, 'FP1=')
            fpVals{1} = extractAfter(line, 'FP1=');
        elseif startsWith(line, 'FP2=')
            fpVals{2} = extractAfter(line, 'FP2=');
        elseif startsWith(line, 'FP3=')
            fpVals{3} = extractAfter(line, 'FP3=');
        end
    end

    fclose(fid);

    % Return as a struct with one field: 'forcePlates'
    fpInfo = struct('forcePlates', fpVals);
end