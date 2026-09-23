## [mu, S] = fr_ogk (X, reweight) — OGK robust location/scatter (Maronna & Zamar 2002) with one
## hard-rejection reweighting at chi2_d(0.975). Mirror of Frontier.jl ogk.
function [mu, S] = fr_ogk (X, reweight)
  if (nargin < 2) reweight = true; endif
  [n, d] = size (X);
  med = median (X, 1);
  s = zeros (1, d);
  for j = 1:d s(j) = fr_rscale (X(:, j)); endfor
  Y = (X - med) ./ s;
  U = eye (d);
  for j = 1:d
    for k = j+1:d
      U(j, k) = (fr_rscale (Y(:, j) + Y(:, k))^2 - fr_rscale (Y(:, j) - Y(:, k))^2) / 4;
      U(k, j) = U(j, k);
    endfor
  endfor
  [E, ~] = eig ((U + U') / 2);
  Z = Y * E;
  g = zeros (d, 1); mz = zeros (d, 1);
  for l = 1:d g(l) = fr_rscale (Z(:, l))^2; mz(l) = median (Z(:, l)); endfor
  D = diag (s);
  S = D * E * diag (g) * E' * D;
  mu = (D * (E * mz))' + med;
  if (reweight && n > d + 1)
    chi2q = @(p) 2 * gammaincinv (p, d / 2);
    R = X - mu;
    d2 = sum ((R / (S + 1e-12 * eye (d))) .* R, 2);
    S = S * median (d2) / chi2q (0.5);
    d2 = sum ((R / (S + 1e-12 * eye (d))) .* R, 2);
    keep = d2 <= chi2q (0.975);
    if (sum (keep) > d + 1)
      mu = mean (X(keep, :), 1);
      S = cov (X(keep, :));
    endif
  endif
  mu = mu(:); S = (S + S') / 2;
endfunction
