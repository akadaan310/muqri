function [score, first, last] = ctc_viterbi(lp, seq, blank)
  % Best CTC path; Octave mirror of CtcGop.ctc_viterbi (Julia). Same tie rule: stay, then s-1,
  % then s-2, a move replacing the incumbent only when strictly better.
  T = rows(lp); L = numel(seq); S = 2 * L + 1;
  ext = repmat(blank, 1, S); ext(2:2:S) = seq;
  skip = false(1, S);
  skip(3:S) = (ext(3:S) ~= blank) & (ext(3:S) ~= ext(1:S-2));
  d = -inf(T, S); bp = zeros(T, S, 'int8');
  d(1, 1) = lp(1, blank); if S > 1, d(1, 2) = lp(1, ext(2)); end
  for t = 2:T
    stay = d(t-1, :);
    p1 = [-inf, d(t-1, 1:S-1)];
    p2 = [-inf, -inf, d(t-1, 1:S-2)]; p2(~skip) = -inf;
    best = stay; arg = zeros(1, S, 'int8');
    k = p1 > best; best(k) = p1(k); arg(k) = 1;
    k = p2 > best; best(k) = p2(k); arg(k) = 2;
    d(t, :) = best + lp(t, ext);
    bp(t, :) = arg;
  end
  s = S; if S > 1 && d(T, S-1) > d(T, S), s = S - 1; end
  score = d(T, s);
  first = zeros(1, L); last = zeros(1, L);
  for t = T:-1:1
    if mod(s, 2) == 0
      k = s / 2;
      if last(k) == 0, last(k) = t; end
      first(k) = t;
    end
    if t > 1, s = s - double(bp(t, s)); end
  end
end
