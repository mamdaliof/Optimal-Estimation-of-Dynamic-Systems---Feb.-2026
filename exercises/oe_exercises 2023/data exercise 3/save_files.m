%% Save all open figures and zip them
output_dir = 'figures_output';
zip_name   = 'C_x=2.zip';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fig_handles = findall(0, 'Type', 'figure');

% Sort by figure number (ascending)
fig_numbers = arrayfun(@(f) f.Number, fig_handles);
[~, sort_idx] = sort(fig_numbers);
fig_handles = fig_handles(sort_idx);

saved_files = {};

for i = 1:numel(fig_handles)
    f   = fig_handles(i);
    num = f.Number;

    % Try to extract a clean title from the axes
    ax = findobj(f, 'Type', 'axes');
    if ~isempty(ax)
        raw_title = ax(1).Title.String;
        % Remove characters that are invalid in filenames
        clean_title = regexprep(raw_title, '[\\/:*?"<>| ]', '_');
        clean_title = regexprep(clean_title, '_+', '_');
        clean_title = strtrim(clean_title);
    else
        clean_title = '';
    end

    if isempty(clean_title)
        filename = sprintf('figure_%02d.png', num);
    else
        filename = sprintf('figure_%02d_%s.png', num, clean_title);
    end

    filepath = fullfile(output_dir, filename);
    exportgraphics(f, filepath, 'Resolution', 300);
    saved_files{end+1} = filepath; %#ok<AGROW>
    fprintf('Saved: %s\n', filepath);
end

zip(zip_name, output_dir);  % <-- zip the whole folder
fprintf('\nAll figures zipped into: %s\n', zip_name);
