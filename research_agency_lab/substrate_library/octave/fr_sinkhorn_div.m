## S = fr_sinkhorn_div (a, b, M, eps) — debiased Sinkhorn divergence (Feydy et al. 2019), log domain.
function S = fr_sinkhorn_div (a, b, M, epsv)
  if (nargin < 4) epsv = 0.05; endif
  S = cost (a, b, M, epsv) - (cost (a, a, M, epsv) + cost (b, b, M, epsv)) / 2;
endfunction
function c = cost (a, b, M, e)
  a = a(:); b = b(:); la = log (max (a, 1e-300)); lb = log (max (b, 1e-300));
  f = zeros (numel (a), 1); g = zeros (numel (b), 1);
  for it = 1:2000
    fo = f;
    f = -e * lse ((g' - M) / e + lb', 2);
    g = -e * lse (((f - M) / e + la)', 2);
    if (max (abs (f - fo)) < 1e-10) break; endif
  endfor
  P = exp ((f + g' - M) / e + la + lb');
  c = sum (P(:) .* M(:));
endfunction
function r = lse (V, dim)
  mx = max (V, [], dim); r = mx + log (sum (exp (V - mx), dim));
endfunction
