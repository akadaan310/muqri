function ll = ctc_forward(lp, seq, blank)
  % log P(seq | lp) over all CTC paths; Octave mirror of CtcGop.ctc_forward (Julia).
  % lp: T x V log-posteriors; seq: 1-based column indices (row vector); blank: blank column.
  T = rows(lp); L = numel(seq);
  if L == 0, ll = sum(lp(:, blank)); return; end
  S = 2 * L + 1;
  ext = repmat(blank, 1, S); ext(2:2:S) = seq;
  skip = false(1, S);
  skip(3:S) = (ext(3:S) ~= blank) & (ext(3:S) ~= ext(1:S-2));
  a = -inf(1, S);
  a(1) = lp(1, blank); a(2) = lp(1, ext(2));
  for t = 2:T
    prev1 = [-inf, a(1:S-1)];
    prev2 = [-inf, -inf, a(1:S-2)];
    prev2(~skip) = -inf;
    m = max([a; prev1; prev2], [], 1);
    fin = isfinite(m);
    v = -inf(1, S);
    v(fin) = m(fin) + log(exp(a(fin) - m(fin)) + exp(prev1(fin) - m(fin)) + exp(prev2(fin) - m(fin)));
    a = v + lp(t, ext);
  end
  m = max(a(S), a(S-1));
  ll = m + log(exp(a(S) - m) + exp(a(S-1) - m));
end
