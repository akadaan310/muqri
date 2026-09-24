% Octave mirror of anatomy.jl: the acts of one consonant (collision, approach, hold, burst, echo).
% x: 16 kHz mono column vector; t0, t1: the consonant's span in seconds. Same 5 ms envelopes, same
% time-domain filters (pre-emphasis 0.97, 20-sample moving average), same windows.
function a = anatomy_acts(x, t0, t1)
  SR = 16000; HOP = 80;
  env = @(v) 10 * log10(sum(reshape(v(1:floor(numel(v)/HOP)*HOP), HOP, []).^2, 1)' / HOP + 1e-12);
  pre = [x(1); x(2:end) - 0.97 * x(1:end-1)];
  c = [0; cumsum(double(x))]; n = numel(x); i = (1:n)';
  low = (c(min(n + 1, i + 20)) - c(i)) / 20;
  e = env(double(x)); h = env(pre); l = env(low);
  win = @(v, s, t) v(max(1, min(numel(v), floor(s * SR / HOP) + 1)):max(1, min(numel(v), floor(t * SR / HOP))));
  prev = win(e, max(0, t0 - 0.06), t0); prev_l = win(l, max(0, t0 - 0.06), t0);
  inside = win(e, t0, t1);
  ref = median(prev); ref_l = median(prev_l);
  entry = win(e, max(0, t0 - 0.03), t0 + 0.03);
  if numel(entry) > 2, approach = max(entry(1:end-2) - entry(3:end)); else, approach = NaN; end
  burst = win(h, max(t0, t1 - 0.03), t1 + 0.03); closure = win(h, t0, t1);
  after = win(e, t1, t1 + 0.12);
  a = [ref - min(inside), approach, median(win(l, t0, t1)) - ref_l, max(burst) - min(closure), max(after) - ref];
end
