## [t, ok] = fr_conformal (scores, alpha) — split-conformal cut: ceil((n+1)(1-alpha))-th order stat.
function [t, ok] = fr_conformal (scores, alpha)
  s = sort (scores(isfinite (scores))); n = numel (s);
  if (n == 0) t = NaN; ok = false; return; endif
  k = ceil ((n + 1) * (1 - alpha));
  ok = k <= n; t = s(min (k, n));
endfunction
