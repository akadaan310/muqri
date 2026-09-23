## [m, S, it] = fr_bw_barycenter (ms, Cs, w) — Bures–Wasserstein barycenter of N(ms{i}, Cs{i}).
## Fixed point S <- S^-1/2 (sum w_i (S^1/2 C_i S^1/2)^1/2)^2 S^-1/2 (Alvarez-Esteban et al. 2016).
function [m, S, it] = fr_bw_barycenter (ms, Cs, w, iters, tol)
  if (nargin < 4) iters = 200; endif
  if (nargin < 5) tol = 1e-12; endif
  w = w(:) / sum (w); k = numel (Cs);
  m = zeros (size (ms{1}(:))); S = zeros (size (Cs{1}));
  for i = 1:k m += w(i) * ms{i}(:); S += w(i) * Cs{i}; endfor
  S = (S + S') / 2;
  for it = 1:iters
    Sh = fr_psd_fun (S, @sqrt); Shi = fr_psd_fun (S, @(x) 1 ./ sqrt (max (x, 1e-300)));
    T = zeros (size (S));
    for i = 1:k T += w(i) * fr_psd_fun (Sh * Cs{i} * Sh, @sqrt); endfor
    Sn = Shi * T * T * Shi; Sn = (Sn + Sn') / 2;
    delta = norm (Sn - S, "fro") / max (norm (S, "fro"), 1e-300);
    S = Sn;
    if (delta < tol) break; endif
  endfor
endfunction
