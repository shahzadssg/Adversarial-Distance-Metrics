%% experiment_adult_income.m
%% Real-world fairness experiment - Adult Income dataset

if ~exist('saved_variables', 'dir')
    mkdir('saved_variables');
end

%% =========================================================================
%% DATA: 10 individuals from Adult Income (5 White, 5 Black)
%% =========================================================================

raw_data = [
    26, 15, 20, 1, 0;
    66,  9, 24, 1, 1;
    45, 14, 40, 1, 1;
    46,  4, 40, 1, 0;
    55, 10, 50, 1, 0;
    81,  5, 16, 0, 0;
    27,  9, 40, 0, 0;
    41, 12, 40, 0, 1;
    33, 12, 55, 0, 0;
    53, 11, 48, 0, 0;
];

race_white = raw_data(:, 4);
income_high = raw_data(:, 5);

features_raw = raw_data(:, 1:3);
mu_feat = mean(features_raw);
sigma_feat = std(features_raw);
y = (features_raw - mu_feat) ./ sigma_feat;

[m, ell] = size(y);

% Unique micro-jitter on data points
for i = 1:m
    y(i, :) = y(i, :) + (i * 1e-8) * ones(1, ell);
end

number_of_classes = 2;

fprintf('\n============================================\n');
fprintf('  Adult Income Experiment (m=%d)\n', m);
fprintf('============================================\n');
fprintf('Features (%d)      : age, education_num, hours_per_week\n', ell);
fprintf('Protected attribute: race (%d White, %d Black)\n', ...
    sum(race_white), sum(~race_white));
fprintf('Ground truth >50K  : %d White, %d Black\n', ...
    sum(race_white & income_high), sum(~race_white & income_high));
fprintf('\n');

%% =========================================================================
%% BASELINE
%% =========================================================================

fprintf('--- Baseline: Euclidean K-means (K=2) ---\n');
[baseline_labels, ~] = kmeans(y, number_of_classes, ...
    'Distance', 'sqeuclidean', 'Replicates', 20);

avg_income = zeros(number_of_classes, 1);
for c = 1:number_of_classes
    avg_income(c) = mean(income_high(baseline_labels == c));
end
[~, favorable_cluster] = max(avg_income);
baseline_favorable = (baseline_labels == favorable_cluster);

p_fav_white_baseline = mean(baseline_favorable(race_white == 1));
p_fav_black_baseline = mean(baseline_favorable(race_white == 0));
dpd_baseline = abs(p_fav_white_baseline - p_fav_black_baseline);

fprintf('P(favorable | White) = %.3f\n', p_fav_white_baseline);
fprintf('P(favorable | Black) = %.3f\n', p_fav_black_baseline);
fprintf('DPD (baseline)       = %.3f\n\n', dpd_baseline);

%% =========================================================================
%% ADVERSARIAL ATTACK
%% =========================================================================

fprintf('--- Adversarial Attack ---\n');

adversarial_labels = zeros(m, 1);
adversarial_labels(race_white == 1) = 1;
adversarial_labels(race_white == 0) = 2;

% Compute cluster centers
k_means_centers = zeros(number_of_classes, ell);
k_means_cluster_counts = zeros(number_of_classes, 1);
for i = 1:m
    ci = adversarial_labels(i);
    k_means_centers(ci, :) = k_means_centers(ci, :) + y(i, :);
    k_means_cluster_counts(ci) = k_means_cluster_counts(ci) + 1;
end
for i = 1:number_of_classes
    k_means_centers(i, :) = k_means_centers(i, :) / k_means_cluster_counts(i);
end

% Augment dataset with cluster centers
y_aug = [k_means_centers; y];
m_aug = size(y_aug, 1);
adv_labels_aug = [(1:number_of_classes)'; adversarial_labels];

% Jitter the center rows in y_aug
for i = 1:number_of_classes
    y_aug(i, :) = y_aug(i, :) + ((m + i) * 1e-8) * ones(1, ell);
end

% *** sync k_means_centers with jittered y_aug ***
k_means_centers = y_aug(1:number_of_classes, :);

% Min/max distances
d_min = Inf; d_max = -Inf;
for i = 1:m_aug
    for j = (i+1):m_aug
        d = norm(y_aug(i,:) - y_aug(j,:), 2);
        d_min = min(d_min, d);
        d_max = max(d_max, d);
    end
end

smallDistance = 0.005 * d_min;
largeDistance = 200 * d_max;

% Desired distances
h = nchoosek(m_aug, 2);
distances = zeros(h, 1);
k = 0;
for i = 1:m_aug
    for j = (i+1):m_aug
        k = k + 1;
        if adv_labels_aug(i) == adv_labels_aug(j)
            distances(k) = smallDistance;
        else
            distances(k) = largeDistance;
        end
        euc_dist = norm(y_aug(i,:) - y_aug(j,:), 2);
        distances(k) = min(distances(k), 0.99 * euc_dist);
    end
end

epsilon = d_min / 10;

fprintf('m_aug=%d, h=%d, ell=%d\n', m_aug, h, ell);
fprintf('Epsilon = %.2e\n\n', epsilon);

fprintf('Constructing epsilon-semimetric...\n');
tic;
crafted_metric = compute_metric(y_aug, distances, epsilon, true, "double");
construction_time = toc;
fprintf('Construction time: %.3f seconds\n', construction_time);

% Diagnostic
load(strcat("saved_variables/As_m", num2str(m_aug), "_double.mat"), "A", "s");
fprintf('Matrix A: norm=%.6e, cond=%.2e\n\n', norm(A,'fro'), condest(A));

%% =========================================================================
%% ADVERSARIAL K-MEANS
%% =========================================================================

fprintf('--- Adversarial K-means ---\n');
tic;
[adv_kmeans_labels, ~] = kmeans_customDist(y_aug, number_of_classes, ...
    crafted_metric, k_means_centers, 1);
kmeans_time = toc;
fprintf('K-means time: %.3f seconds\n', kmeans_time);

adv_result = adv_kmeans_labels(number_of_classes+1:end);

perms_mat = perms(1:number_of_classes);
best_acc = 0; best_perm = [];
for p = 1:size(perms_mat, 1)
    perm = perms_mat(p, :);
    permuted = arrayfun(@(x) perm(x), adv_result);
    acc = mean(permuted == adversarial_labels);
    if acc > best_acc
        best_acc = acc; best_perm = perm;
    end
end

adv_mapped = arrayfun(@(x) best_perm(x), adv_result);
fprintf('K-means attack accuracy: %.1f%%\n', best_acc * 100);

adv_favorable = (adv_mapped == 1);
p_fav_white_adv = mean(adv_favorable(race_white == 1));
p_fav_black_adv = mean(adv_favorable(race_white == 0));
dpd_adv = abs(p_fav_white_adv - p_fav_black_adv);

fprintf('P(favorable | White) = %.3f\n', p_fav_white_adv);
fprintf('P(favorable | Black) = %.3f\n', p_fav_black_adv);
fprintf('DPD (adversarial)    = %.3f\n\n', dpd_adv);

% Distance diagnostic
fprintf('--- Distance diagnostic ---\n');
fprintf('%-6s %-6s %-14s %-14s %-8s\n', 'Point', 'Race', 'd(own_ctr)', 'd(other_ctr)', 'Correct?');
for i = 1:m
    idx = i + number_of_classes;
    d_to_c1 = crafted_metric(y_aug(idx,:), y_aug(1,:));
    d_to_c2 = crafted_metric(y_aug(idx,:), y_aug(2,:));
    own_cluster = adversarial_labels(i);
    if own_cluster == 1
        d_own = d_to_c1; d_other = d_to_c2;
    else
        d_own = d_to_c2; d_other = d_to_c1;
    end
    correct = real(d_own) < real(d_other);
    race_str = 'W';
    if race_white(i) == 0; race_str = 'B'; end
    fprintf('%-6d %-6s %-14.6f %-14.6f %-8s\n', i, race_str, ...
        real(d_own), real(d_other), string(correct));
end

%% =========================================================================
%% ADVERSARIAL DBSCAN
%% =========================================================================

fprintf('\n--- Adversarial DBSCAN ---\n');
tic;
[adv_dbscan_labels, ~] = dbscan_customDist(y_aug, 2, ...
    0.5*(smallDistance + largeDistance), crafted_metric);
dbscan_time = toc;
fprintf('DBSCAN time: %.3f seconds\n', dbscan_time);

adv_dbscan_result = adv_dbscan_labels(number_of_classes+1:end);
if min(adv_dbscan_result) == 0
    adv_dbscan_result = adv_dbscan_result + 1;
end

best_acc_db = 0;
for p = 1:size(perms_mat, 1)
    perm = perms_mat(p, :);
    permuted = arrayfun(@(x) perm(x), adv_dbscan_result);
    acc = mean(permuted == adversarial_labels);
    best_acc_db = max(best_acc_db, acc);
end
fprintf('DBSCAN attack accuracy: %.1f%%\n\n', best_acc_db * 100);

%% =========================================================================
%% SUMMARY
%% =========================================================================

fprintf('============================================\n');
fprintf('  SUMMARY\n');
fprintf('============================================\n');
fprintf('%-30s %-12s %-12s\n', '', 'Baseline', 'Adversarial');
fprintf('------------------------------------------------------------\n');
fprintf('%-30s %-12.3f %-12.3f\n', 'P(favorable | White)', ...
    p_fav_white_baseline, p_fav_white_adv);
fprintf('%-30s %-12.3f %-12.3f\n', 'P(favorable | Black)', ...
    p_fav_black_baseline, p_fav_black_adv);
fprintf('%-30s %-12.3f %-12.3f\n', 'DPD', dpd_baseline, dpd_adv);
fprintf('%-30s %-12s %-12.1f%%\n', 'K-means attack accuracy', '---', best_acc * 100);
fprintf('%-30s %-12s %-12.1f%%\n', 'DBSCAN attack accuracy', '---', best_acc_db * 100);
fprintf('%-30s %-12s %-12.3f s\n', 'Construction time', '---', construction_time);
fprintf('%-30s %-12s %-12.2e\n', 'Epsilon', '---', epsilon);
fprintf('============================================\n');