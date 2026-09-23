## M = fr_karcher (Cs, w) — affine-invariant Frechet mean, started at the log-Euclidean mean.
function M = fr_karcher (Cs, w, iters, tol)
  if (nargin < 3) iters = 100; endif
  if (nargin < 4) tol = 1e-12; endif
  w = w(:) / sum (w); k = numel (Cs);
  L = zeros (size (Cs{1}));
  for i = 1:k L += w(i) * fr_psd_fun (Cs{i}, @(x) log (max (x, 1e-300))); endfor
  M = expm ((L + L') / 2);
  for t = 1:iters
    Mh = fr_psd_fun (M, @sqrt); Mhi = fr_psd_fun (M, @(x) 1 ./ sqrt (max (x, 1e-300)));
    G = zeros (size (M));
    for i = 1:k G += w(i) * fr_psd_fun (Mhi * Cs{i} * Mhi, @(x) log (max (x, 1e-300))); endfor
    M = Mh * expm ((G + G') / 2) * Mh; M = (M + M') / 2;
    if (norm (G, "fro") < tol) break; endif
  endfor
endfunction
