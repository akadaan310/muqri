## q = fr_quantile7 (x, p) — Hyndman–Fan type 7 quantile (Julia/StatsBase and R default).
function q = fr_quantile7 (x, p)
  x = sort (x(:)); n = numel (x);
  h = (n - 1) * p(:) + 1; lo = floor (h); hi = min (lo + 1, n);
  q = x(lo) + (h - lo) .* (x(hi) - x(lo));
endfunction
