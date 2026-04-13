%% Parameters
%% x, y: two points in the (low-dimensional) space
%% s: scaling factor for the (deterministic) noise
%% A: matrix to define the quadratic form (must be square)
function d = epsilon_semimetric(x, y, s, A)
ex = size(A,1) - max(size(x));
%% initialize the pseudorandom noise

seed1 = abs(prod(x)*sum(y));
seed1 = int32(seed1 / 10^floor(log10(seed1)) * 1e8);
rng(seed1);
%rand("seed", prod(x)*sum(y));  %% let the seed depend on *both* x and y
zi = [x, s * rand([1 ex])];

seed2 = abs(prod(y)*sum(x));
seed2 = int32(seed2 / 10^floor(log10(seed2)) * 1e8);
rng(seed2);
%rand("seed", prod(y)*sum(x));  %% let the seed depend on *both* x and y
zj = [y, s * rand([1 ex])];
%rand("seed", "reset");  %% avoid unwanted side-effects (for other routines)
v = zi - zj;
d = sqrt(v*A*v');
end