function y = px_z(x, z)
    % Posterior p(x|z) = p(z|x) p(x) / p(z) via Bayes rule
    y = (pz_x(z, x) .* px(x)) ./ pz(z);
end
