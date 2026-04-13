function metric = compute_metric(y, distances, epsilon, print_distances, precision)
%COMPUTE_METRIC Construct epsilon semimetric based on data and distances

%% Lade das symbolische Paket fuer Octave
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if (isOctave)
   %setenv PYTHON python
   pkg load symbolic
end

if nargin < 5
  precision="double";
end
if nargin < 4
  print_distances=false;
end


[m, ell] = size(y);
h = nchoosek(m, 2);
ex = h - ell;

M = zeros([h,h]);
s = 0.9 * epsilon / sqrt(h); %% scaling factor for the noise, so that the noise-points are less than epsilon far from the original points

k = 0;
for i = 1:m
    for j = (i+1):m
        k = k + 1;

        seed1 = abs(prod(y(i,:))*sum(y(j,:))); % let the seed depend on *both* y(i) and y(j)
        seed1 = int32(seed1 / 10^floor(log10(seed1)) * 1e8); % seed needs to be a nonnegative integer; it is scaled to not lose info before type conversion
        rng(seed1);
        zi = [y(i,:), s * rand([1 ex])];  %% now, the sequence is pseudorandom

        seed2 = abs(prod(y(j,:))*sum(y(i,:)));
        seed2 = int32(seed2 / 10^floor(log10(seed2)) * 1e8);
        rng(seed2);
        zj = [y(j,:), s * rand([1 ex])];

        M(:,k) = (zi - zj)';
    end
end

[Q_full, R_full] = qr(M);

A = zeros(h, h);

for i = 1:h
    [Q, ~] = qrdelete(Q_full,R_full,i);
    if (strcmp(precision, "double"))
        B = Q(:,end)';
    elseif (strcmp(precision, "variable"))
        B = vpa(Q(:,end)');
    end

    A_i = B'*B;

    lambda = M(:,i)' * A_i * M(:,i);
    A = A + A_i / lambda * distances(i)^2;
end

if print_distances
    crafted_norm = @(y) (sqrt(y'*A*y));

    %test if the quadratic form-norm delivers the sought distances
    disp(['point pair  ' 'desired  ' 'computed distance' ])
    for i = 1:h
        disp([i distances(i) crafted_norm(M(:,i))])
    end
end

save(strcat("saved_variables/As_m", num2str(m), "_", precision, ".mat"),"A","s");

metric = @(y,y_prime) (epsilon_semimetric(y, y_prime, s, A));

end