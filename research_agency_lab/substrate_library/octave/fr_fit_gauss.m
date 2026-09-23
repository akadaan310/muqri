## [m, C] = fr_fit_gauss (Z) — N(m, C) in whitened space, shrunk (n C + k I)/(n + k), k = d + 1.
function [m, C] = fr_fit_gauss (Z)
  [n, d] = size (Z);
  if (n >= 2 * d + 3)
    [m, C] = fr_ogk (Z);
  else
    m = median (Z, 1)'; if (n > 1) C = cov (Z); else C = zeros (d); endif
  endif
  k = d + 1; C = (n * C + k * eye (d)) / (n + k); C = (C + C') / 2;
endfunction
