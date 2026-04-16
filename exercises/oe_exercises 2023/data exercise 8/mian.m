clc; close all; clear
%% Part I - Q1

data =load('SLAM.mat'); 

F = eye(2);

disp('System matrix F:');
disp(F);

%% Q2

C_w_tilde = diag([data.std_velocity^2, data.std_heading^2]);
disp('Raw noise covariance C_w_tilde:');
disp(C_w_tilde);

G = @(v, phi) data.delta * ...
    [ cos(phi * pi/180),   -v * sin(phi * pi/180) * (pi/180) ; ...
      sin(phi * pi/180),    v * cos(phi * pi/180) * (pi/180) ];
C_w = @(v, phi) G(v, phi) * C_w_tilde * G(v, phi)';


%% Q3

function plot_ellipse(center, P, color)
    [V, D] = eig(P);                   
    theta  = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    ellipse = center + 2 * V * sqrt(D) * circle;
    plot(ellipse(1,:), ellipse(2,:), color, 'LineWidth', 1.5);
end

N       = size(data.u, 2);  
pos     = [0; 0];         
cov_pos = zeros(2);        

track      = zeros(2, N+1);
track(:,1) = pos;

figure; hold on; grid on; axis equal;
title('Part I: Predicted Trajectory with Uncertainty Regions');
xlabel('x [m]'); ylabel('y [m]');
plot(0, 0, 'go', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start');

for i = 1:N
    v_i   = data.u(1, i);
    phi_i = data.u(2, i);

    pos = F * pos + data.delta * [ cos(phi_i*pi/180) * v_i ;
                              sin(phi_i*pi/180) * v_i ];
    cov_pos = F * cov_pos * F' + C_w(v_i, phi_i);

    track(:, i+1) = pos;

    if mod(i, 50) == 0
        plot_ellipse(pos, cov_pos, 'r-');
        plot(pos(1), pos(2), 'r.', 'MarkerSize', 15);
       
    end
end

plot(track(1,:), track(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Predicted track');
legend('Start', 'Uncertainty (2\sigma)', 'Predicted track', 'Location', 'best');
hold off

%% Part II - Q7, Q8, Q9

X = [0; 0];         
C = zeros(2);       
known_ids = [];     
track_slam = zeros(2, N+1);
track_slam(:,1) = X(1:2);

X_saved = cell(1, N); 
C_saved = cell(1, N);
X_pred_saved = cell(1, N); 
C_pred_saved = cell(1, N);

video_filename = 'SLAM_Simulation.avi';
v = VideoWriter(video_filename, 'Motion JPEG AVI');
v.FrameRate = 15;
open(v);
snapshot_steps = round([0.25*N, 0.50*N, 0.75*N, N]);
snapshot_count = 1;

fig_slam = figure('Name', 'SLAM Visualization'); 
hold on; grid on; axis equal;
title('Part II: EKF SLAM');
xlabel('x [m]'); ylabel('y [m]');
plot(0, 0, 'go', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start');
p_track = plot(X(1), X(2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'SLAM track');

for i = 1:N
    %Prediction  
    v_i   = data.u(1, i);
    phi_i = data.u(2, i);
    
    X(1:2) = X(1:2) + data.delta * [cos(phi_i*pi/180) * v_i; 
                                    sin(phi_i*pi/180) * v_i];
    
    L = length(X);
    F_slam = eye(L);
    
    C_w_slam = zeros(L, L);
    C_w_slam(1:2, 1:2) = C_w(v_i, phi_i); 
    
    C = F_slam * C * F_slam' + C_w_slam;

    X_pred_saved{i} = X;
    C_pred_saved{i} = C;

    % Observation  
    if ~isempty(data.Z{i}.id)
        visible_ids = data.Z{i}.id;
        z_pos_measurements = data.Z{i}.zpos;
        
        for m = 1:length(visible_ids)
            lm_id = visible_ids(m);
            
            if ~ismember(lm_id, known_ids)
                % New landmark
                known_ids = [known_ids, lm_id];
                X = [X; X(1:2)]; 
                
                C_new = eye(2) * 1e6; 
                C = blkdiag(C, C_new);
            end
        end
        
        % Update  
        L = length(X); 
        num_vis = length(visible_ids);
        
        z = zeros(2*num_vis, 1);
        z_hat = zeros(2*num_vis, 1);
        H = zeros(2*num_vis, L);
        C_n = eye(2*num_vis) * (data.stdn^2); 
        
        for m = 1:num_vis
            lm_id = visible_ids(m);
            
            idx = find(known_ids == lm_id);
            state_idx = 2 + 2*idx - 1; 
            
            % Actual measurement
            z(2*m-1 : 2*m) = z_pos_measurements(:, m);
            
            % Predicted measurement
            z_hat(2*m-1 : 2*m) = X(state_idx : state_idx+1) - X(1:2);
            
            H(2*m-1 : 2*m, 1:2) = [-1, 0; 0, -1];
            H(2*m-1 : 2*m, state_idx : state_idx+1) = [1, 0; 0, 1]; 
        end
        
        y = z - z_hat; 
        S = H * C * H' + C_n;
        K = C * H' / S;
        
        X = X + K * y;
        C = (eye(L) - K * H) * C;
    end
    
    track_slam(:, i+1) = X(1:2);
    X_saved{i} = X;
    C_saved{i} = C;

  % Visualization
    if mod(i, 5) == 0 || i == N 
        set(p_track, 'XData', track_slam(1, 1:i+1), 'YData', track_slam(2, 1:i+1));
        delete(findobj(gcf, 'Tag', 'dynamic_ellipse'));
        
        % Plot Vehicle
        [V_eig, D_eig] = eig(C(1:2, 1:2));                   
        theta  = linspace(0, 2*pi, 100);
        circle = [cos(theta); sin(theta)];
        ellipse = X(1:2) + 2 * V_eig * sqrt(D_eig) * circle;
        plot(ellipse(1,:), ellipse(2,:), 'r-', 'LineWidth', 1.5, 'Tag', 'dynamic_ellipse');
        
        % Plot Landmarks
        for m = 1:length(known_ids)
            state_idx = 2 + 2*m - 1;
            plot(X(state_idx), X(state_idx+1), 'k.', 'MarkerSize', 3, 'Tag', 'dynamic_ellipse'); 
            
            [V_eig, D_eig] = eig(C(state_idx:state_idx+1, state_idx:state_idx+1));
            ellipse = X(state_idx:state_idx+1) + 2 * V_eig * sqrt(D_eig) * circle;
            plot(ellipse(1,:), ellipse(2,:), 'k-', 'LineWidth', 1.5, 'Tag', 'dynamic_ellipse');
        end
        
        drawnow; 
        
       % Q8: Capture video frame
        vidFrame = getframe(fig_slam); 
        
        % On the first frame, record the exact pixel dimensions
        if ~exist('vid_frame_size', 'var')
            vid_frame_size = size(vidFrame.cdata);
        else
            % Force all subsequent frames to match the exact size of the first frame
            vidFrame.cdata = imresize(vidFrame.cdata, [vid_frame_size(1), vid_frame_size(2)]);
        end
        
        writeVideo(v, vidFrame);
        
        % Q8: Save 4 snapshot figures
        if ismember(i, snapshot_steps)
            filename = sprintf('SLAM_Trajectory_Step_%d.png', snapshot_count);
            exportgraphics(fig_slam, filename, 'Resolution', 300);
            disp(['Saved snapshot: ', filename]);
            snapshot_count = snapshot_count + 1;
        end
    end
end
close(v);
disp(['Video saved to ', video_filename]);
legend('Start', 'SLAM track', 'Location', 'best');

figure('Name', 'Comparison: SLAM vs Prediction'); 
hold on; grid on; axis equal;
title('Part II (Q9): SLAM vs Prediction Only');
xlabel('x [m]'); ylabel('y [m]');
% Plot Prediction (from Part I code - assumes 'track' exists in workspace)
plot(track(1,:), track(2,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Prediction Only (Dead Reckoning)');
% Plot SLAM (from Part II code)
plot(track_slam(1,:), track_slam(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'SLAM Track');
plot(0, 0, 'go', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start');


% Plot final estimated landmarks for context
for m = 1:length(known_ids)
    state_idx = 2 + 2*m - 1;
    plot(X_saved{end}(state_idx), X_saved{end}(state_idx+1), 'k*', 'MarkerSize', 5, 'HandleVisibility', 'off');
end
legend('Location', 'best');
exportgraphics(gcf, 'Q9_SLAM_vs_Prediction_Comparison.png', 'Resolution', 300);
disp('Saved comparison plot: Q9_SLAM_vs_Prediction_Comparison.png');

%% Part III 

X_smooth = cell(1, N);
C_smooth = cell(1, N);

X_smooth{N} = X_saved{N};
C_smooth{N} = C_saved{N};

track_smooth = zeros(2, N);
track_smooth(:, N) = X_smooth{N}(1:2);

% Backward pass
for j = N-1 : -1 : 1
    
    len_x = length(X_saved{j});    
    F = eye(len_x);
    
    X_pred_P = X_pred_saved{j+1}(1:len_x);
    C_pred_P = C_pred_saved{j+1}(1:len_x, 1:len_x);
    X_smooth_next_P = X_smooth{j+1}(1:len_x);
    C_smooth_next_P = C_smooth{j+1}(1:len_x, 1:len_x);
   
    B = C_saved{j} * F' / C_pred_P; 
    X_smooth{j} = X_saved{j} + B * (X_smooth_next_P - X_pred_P);
    C_smooth{j} = C_saved{j} + B * (C_smooth_next_P - C_pred_P) * B';
    track_smooth(:, j) = X_smooth{j}(1:2);
end

disp('RTS Smoothing completed.');


% --- Create Smoothed Movie ---
video_smooth = 'SLAM_Smoothed_Simulation.avi';
v_smooth = VideoWriter(video_smooth, 'Motion JPEG AVI');
v_smooth.FrameRate = 15;
open(v_smooth);

fig_smooth = figure('Name', 'RTS Smoother Visualization');
hold on; grid on; axis equal;
title('Part III: RTS Smoothed SLAM (Backward Animation)');
xlabel('x [m]'); ylabel('y [m]');

% Plot the static background elements
plot(0, 0, 'go', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start');
plot(track_slam(1,:), track_slam(2,:), 'b--', 'LineWidth', 1, 'DisplayName', 'Forward SLAM Track');

% Initialize the smoothed track at the final position (N)
p_smooth_track = plot(track_smooth(1,N), track_smooth(2,N), 'g-', 'LineWidth', 2, 'DisplayName', 'Smoothed Track');

% Create dummy plots strictly to add the ellipses to the legend properly
plot(NaN, NaN, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Forward Covariance');
plot(NaN, NaN, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Smoothed Covariance');
legend('Location', 'best');

% Pre-compute circle for efficiency
theta  = linspace(0, 2*pi, 100);
circle = [cos(theta); sin(theta)];

% --- NEW: Initialize empty objects for the CURRENT position and ellipse ---
% This guarantees we only ever show one at a time without needing delete()
p_fwd_pos = plot(NaN, NaN, 'k.', 'MarkerSize', 10, 'HandleVisibility', 'off');
p_fwd_ell = plot(NaN, NaN, 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');

p_sm_pos = plot(NaN, NaN, 'r.', 'MarkerSize', 10, 'HandleVisibility', 'off');
p_sm_ell = plot(NaN, NaN, 'r-', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% Run the loop backwards from N down to 1
for i = N:-5:1 
    % Update the drawn smoothed line from the current backward step (i) to the end (N)
    set(p_smooth_track, 'XData', track_smooth(1, i:N), 'YData', track_smooth(2, i:N));
    
    % 1. Update Current Forward SLAM Vehicle Location and Covariance (Black)
    [V_fwd, D_fwd] = eig(C_saved{i}(1:2, 1:2));                   
    ellipse_fwd = X_saved{i}(1:2) + 2 * V_fwd * sqrt(D_fwd) * circle;
    set(p_fwd_pos, 'XData', X_saved{i}(1), 'YData', X_saved{i}(2));
    set(p_fwd_ell, 'XData', ellipse_fwd(1,:), 'YData', ellipse_fwd(2,:));
    
    % 2. Update Current RTS Smoothed Vehicle Location and Covariance (Red)
    [V_sm, D_sm] = eig(C_smooth{i}(1:2, 1:2));                   
    ellipse_sm = X_smooth{i}(1:2) + 2 * V_sm * sqrt(D_sm) * circle;
    set(p_sm_pos, 'XData', X_smooth{i}(1), 'YData', X_smooth{i}(2));
    set(p_sm_ell, 'XData', ellipse_sm(1,:), 'YData', ellipse_sm(2,:));
    
    drawnow;
    
    % Capture frame safely
    vidFrame = getframe(fig_smooth); 
    if ~exist('vid_frame_size_sm', 'var')
        vid_frame_size_sm = size(vidFrame.cdata);
    else
        vidFrame.cdata = imresize(vidFrame.cdata, [vid_frame_size_sm(1), vid_frame_size_sm(2)]);
    end
    writeVideo(v_smooth, vidFrame);
end
close(v_smooth);
disp(['Smoothed video saved to ', video_smooth]);

% --- Final Comparison Graph ---
figure('Name', 'Final Comparison: SLAM vs RTS');
hold on; grid on; axis equal;
title('Part III: Forward SLAM vs. RTS Smoother');
xlabel('x [m]'); ylabel('y [m]');

plot(track_slam(1,:), track_slam(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Forward EKF SLAM');
plot(track_smooth(1,:), track_smooth(2,:), 'g-', 'LineWidth', 2, 'DisplayName', 'RTS Smoother');

% Plot final smoothed landmarks
final_state = X_smooth{N};
for m = 1:(length(final_state)-2)/2
    state_idx = 2 + 2*m - 1;
    plot(final_state(state_idx), final_state(state_idx+1), 'k*', 'MarkerSize', 5, 'HandleVisibility', 'off');
end

legend('Location', 'best');
exportgraphics(gcf, 'Q10_SLAM_vs_Smoother_Comparison.png', 'Resolution', 300);
disp('Saved comparison plot: Q10_SLAM_vs_Smoother_Comparison.png');

%% Part IV - Q12, Q13, Q14
std_b = 2;         
X = [0; 0];         
C = zeros(2);       
known_ids = [];     
track_slam_bearing = zeros(2, N+1);
track_slam_bearing(:,1) = X(1:2);
X_saved = cell(1, N); 
C_saved = cell(1, N);
X_pred_saved = cell(1, N); 
C_pred_saved = cell(1, N);

video_filename = 'BearingOnly_SLAM_Simulation.avi'; 
v = VideoWriter(video_filename, 'Motion JPEG AVI');
v.FrameRate = 15;
open(v);
snapshot_steps = round([0.25*N, 0.50*N, 0.75*N, N]);
snapshot_count = 1;

fig_slam = figure('Name', 'SLAM Visualization'); 
hold on; grid on; axis equal;
title('Part IV: Bearing-Only EKF SLAM'); 
xlabel('x [m]'); ylabel('y [m]');
plot(0, 0, 'go', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start');
p_track = plot(X(1), X(2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'SLAM track');

for i = 1:N
    %Prediction  
    v_i   = data.u(1, i);
    phi_i = data.u(2, i);
    
    X(1:2) = X(1:2) + data.delta * [cos(phi_i*pi/180) * v_i; 
                                    sin(phi_i*pi/180) * v_i];
    
    L = length(X);
    F_slam = eye(L);
    
    C_w_slam = zeros(L, L);
    C_w_slam(1:2, 1:2) = C_w(v_i, phi_i); 
    
    C = F_slam * C * F_slam' + C_w_slam;
    X_pred_saved{i} = X;
    C_pred_saved{i} = C;
    
    % Observation  
    if ~isempty(data.Z{i}.id)
        visible_ids = data.Z{i}.id;
        

        z_bearing_measurements = data.Z{i}.zbearing; 
        
        for m = 1:length(visible_ids)
            lm_id = visible_ids(m);
            
            if ~ismember(lm_id, known_ids)
                % New landmark
                known_ids = [known_ids, lm_id];
                
                theta_meas = z_bearing_measurements(m);
                x_new = X(1) + 22 * cos(theta_meas * pi / 180);
                y_new = X(2) + 22 * sin(theta_meas * pi / 180);
                
                X = [X; x_new; y_new]; 
                
                C_new = eye(2) * 100; 
                C = blkdiag(C, C_new);
            end
        end
        
        % Update  
        L = length(X); 
        num_vis = length(visible_ids);
        
        z = zeros(num_vis, 1);       
        z_hat = zeros(num_vis, 1);   
        H = zeros(num_vis, L);       
        C_n = eye(num_vis) * (std_b^2); 
        
        for m = 1:num_vis
            lm_id = visible_ids(m);
            
            idx = find(known_ids == lm_id);
            state_idx = 2 + 2*idx - 1; 
            
            z(m) = z_bearing_measurements(m);
            
            dx = X(state_idx) - X(1);
            dy = X(state_idx+1) - X(2);
            r2 = dx^2 + dy^2;
            
            z_hat(m) = atan2(dy, dx) * 180 / pi;
            
            H(m, 1) = (180/pi) * (dy / r2);               
            H(m, 2) = (180/pi) * (-dx / r2);              
            H(m, state_idx) = (180/pi) * (-dy / r2);      
            H(m, state_idx+1) = (180/pi) * (dx / r2);     
        end
        
        y = z - z_hat; 
        y = mod(y + 180, 360) - 180; 
        
        S = H * C * H' + C_n;
        K = C * H' / S;
        
        X = X + K * y;
        C = C - K * H * C; % CHANGED: Kept the optimized standard form from earlier
    end
    
    track_slam_bearing(:, i+1) = X(1:2);
    X_saved{i} = X;
    C_saved{i} = C;
    
  % Visualization
    if mod(i, 50) == 0 || i == N 
        set(p_track, 'XData', track_slam_bearing(1, 1:i+1), 'YData', track_slam_bearing(2, 1:i+1));
        delete(findobj(gcf, 'Tag', 'dynamic_ellipse'));
        
        % Plot Vehicle
        [V_eig, D_eig] = eig(C(1:2, 1:2));                   
        theta  = linspace(0, 2*pi, 100);
        circle = [cos(theta); sin(theta)];
        ellipse = X(1:2) + 2 * V_eig * sqrt(D_eig) * circle;
        plot(ellipse(1,:), ellipse(2,:), 'r-', 'LineWidth', 1.5, 'Tag', 'dynamic_ellipse');
        
        % Plot Landmarks
        for m = 1:length(known_ids)
            state_idx = 2 + 2*m - 1;
            plot(X(state_idx), X(state_idx+1), 'k.', 'MarkerSize', 3, 'Tag', 'dynamic_ellipse'); 
            
            [V_eig, D_eig] = eig(C(state_idx:state_idx+1, state_idx:state_idx+1));
            ellipse = X(state_idx:state_idx+1) + 2 * V_eig * sqrt(D_eig) * circle;
            plot(ellipse(1,:), ellipse(2,:), 'k-', 'LineWidth', 1.5, 'Tag', 'dynamic_ellipse');
        end
        
        drawnow; 
        
       % Capture video frame
        vidFrame = getframe(fig_slam); 
        
        if ~exist('vid_frame_size_b', 'var') % CHANGED: renamed variable slightly to avoid workspace conflict
            vid_frame_size_b = size(vidFrame.cdata);
        else
            vidFrame.cdata = imresize(vidFrame.cdata, [vid_frame_size_b(1), vid_frame_size_b(2)]);
        end
        
        writeVideo(v, vidFrame);
        
        % Save 4 snapshot figures
        if ismember(i, snapshot_steps)
            filename = sprintf('Bearing_SLAM_Step_%d.png', snapshot_count); % CHANGED: filename
            exportgraphics(fig_slam, filename, 'Resolution', 300);
            disp(['Saved snapshot: ', filename]);
            snapshot_count = snapshot_count + 1;
        end
    end
end
close(v);
disp(['Video saved to ', video_filename]);

% --- CHANGED: Final Comparison Plot Details ---
figure('Name', 'Comparison: Bearing SLAM vs SLAM vs Prediction'); 
hold on; grid on; axis equal;
title('Part IV (Q14): Bearing-Only SLAM vs SLAM vs Prediction');
xlabel('x [m]'); ylabel('y [m]');

% Plot Prediction (from Part I code - assumes 'track' exists in workspace)
plot(track(1,:), track(2,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Prediction Only (Dead Reckoning)');
% Plot SLAM (from Part IV code)
plot(track_slam_bearing(1,:), track_slam_bearing(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Bearing-Only SLAM Track');
plot(track_slam(1,:), track_slam(2,:), 'LineWidth', 1.5, 'DisplayName',  'SLAM Track');

plot(0, 0, 'go', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start');

% Plot final estimated landmarks for context
for m = 1:length(known_ids)
    state_idx = 2 + 2*m - 1;
    plot(X_saved{end}(state_idx), X_saved{end}(state_idx+1), 'k*', 'MarkerSize', 5, 'HandleVisibility', 'off');
end
legend('Location', 'best');
exportgraphics(gcf, 'Q14_Bearing_SLAM_vs_Prediction.png', 'Resolution', 300);
disp('Saved comparison plot: Q14_Bearing_SLAM_vs_Prediction.png');