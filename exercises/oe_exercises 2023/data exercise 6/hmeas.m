function z = hmeas(x,x0,Cv)
% z = hmeas(x,x0) measurement function of a yacht
%
% input:
% x: 8D current state vector consisting of:
%    - postition (2D)
%    - velocity (2D)
%    - acceleration (2D)
%    - actual trust (1D)
%    - actual heading (1D)
% x0:  beacon postion
% output:
% z: 3D measurement vector consisting of:
%    - bearing to beacon x0 (degrees)
%    - speed
%    - heading (direction of velocity in degrees)
%
% optional:
% z = hmeas(x,x0,Cv) calculates the measurement with addition of white process
% noise with covariance matrix Cv. This 3x3 matrix can be any valid covariance
% matrix.
% Note: if x is an 8xN matrix, then each column in x is regarded as a state
% vector, and the resulting measurements are calculated in parallel. That
% is, z will  be a 3xN matrix.


z = zeros(3,size(x,2));
z(1,:) = (180/pi)*atan2(x0(2)-x(2,:),x0(1)-x(1,:));
z(2,:) = (x(3,:).^2 + x(4,:).^2).^0.5;
z(3,:) = (180/pi)*atan2(x(4,:),x(3,:));


if nargin>2
    z = z + Cv^0.5 * randn(size(z));
end
