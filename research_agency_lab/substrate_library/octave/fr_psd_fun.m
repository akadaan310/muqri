## F = fr_psd_fun (C, f) — apply f to the eigenvalues of symmetric PSD C (negatives clamped to 0).
## Octave mirror of QaariLab.psd_fun (Frontier.jl).
function F = fr_psd_fun (C, f)
  C = (C + C') / 2;
  [V, L] = eig (C);
  l = max (diag (L), 0);
  F = V * diag (f (l)) * V';
  F = (F + F') / 2;
endfunction
