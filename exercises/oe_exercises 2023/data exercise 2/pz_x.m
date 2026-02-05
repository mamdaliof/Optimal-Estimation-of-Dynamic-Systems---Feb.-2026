function y = pz_x(z,x)

P0 = 0.95;    
P1 = 0.05;    
sigma = 0.1;    

g1 = (1./(sqrt(2*pi)*sigma)) .* exp( -(z - x).^2 ./ (2*sigma^2) );
g2 = (1./(sqrt(2*pi)*sigma)) .* exp( -(z - 2*x).^2 ./ (2*sigma^2) );

y = P0.*g1 + P1.*g2;
end