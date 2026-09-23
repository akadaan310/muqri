function [a, e] = lpc(x, p)
% LPC  Linear predictive coefficients (autocorrelation method).
%
%   [a, e] = lpc(x, p)
%
% Compatibility shim: the Debian ``octave-signal`` 1.4.1 package ships ``levinson`` but not
% ``lpc``. This reproduces the standard (and MATLAB/Octave-signal) definition exactly: the
% biased autocorrelation of ``x`` up to lag ``p`` solved by Levinson-Durbin recursion. ``a`` is
% the length ``p+1`` prediction polynomial (``a(1) == 1``); ``e`` is the final prediction-error
% power. The autocorrelation scale cancels in the recursion, so ``a`` matches ``lpc`` bit-for-bit
% up to floating point.
  x = x(:);
  N = numel(x);
  if nargin < 2 || isempty(p)
    p = 1;
  end
  p = min(p, N - 1);
  R = zeros(p + 1, 1);
  for k = 0:p
    R(k + 1) = sum(x(1:N - k) .* x(k + 1:N));   % biased autocorrelation, lag k
  end
  if R(1) == 0
    a = [1, zeros(1, p)];
    e = 0;
    return;
  end
  [a, e] = levinson(R, p);   % a: row, a(1)=1; e: error power
end
