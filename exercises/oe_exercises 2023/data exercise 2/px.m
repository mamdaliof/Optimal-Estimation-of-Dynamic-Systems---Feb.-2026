function y = px(x)

xmin = 1;    
xmax = 3;    
beta = 20;

mu   = (xmin + xmax)/2;  
alpha = (xmax - xmin)/2;  

y = (beta./(2*alpha * gamma(1/beta))).*exp( -abs((x - mu)/alpha).^beta );
end
