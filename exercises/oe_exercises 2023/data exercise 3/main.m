%% run_all.m  –  launcher: captures diary then zips everything
output_dir = 'figures_output';
diary_file = fullfile(output_dir, 'command_output.txt');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Start capturing Command Window output
diary(diary_file);
diary on;

% ── Run your main script ──────────────────────────────────────
run('Q1to5.m');   % <-- replace with your actual script name
% ─────────────────────────────────────────────────────────────

diary off;

% Now zip figures + diary
run('save_files.m');
