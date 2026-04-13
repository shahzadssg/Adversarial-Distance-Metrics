%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU Affero General Public License as
%    published by the Free Software Foundation, either version 3 of the
%    License, or any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU Affero General Public License for more details. It can be found
%    at <http://www.gnu.org/licenses/>.
%
%==========================================================================
% FUNCTION: [assignments, centers] = kmeans(X, k, centers = 0, maxiter = 200)
% DESCRIPTION: This function performs k-means clustering algorithm on a given
%              dataset.
%
% INPUTS:   X = N*d matrix of dataset, rows of X correspond to N data points;
%               columns correspond to attributes.
%           k = number of clusters
%     centers = Starting centers of clusters.
%     maxiter = Maximum iteration count for convergence.
%
% OUTPUTS: assignments = Integer vector that holds
%==========================================================================
% copyright (c) 2010 M. Emin Aksehirli
%==========================================================================
function  [assignments, centers] = kmeans_customDist(X, k, custom_metric, centers, maxiter)
if (centers == 0)
    centerRows = randperm(size(X,1));
    centers = X(centerRows(1:k), :);
end
numOfRows = length(X(:,1));
numOfFeatures = length(X(1,:));
assignments = ones(1, numOfRows);

for iter = 1:maxiter
    clusterTotals = zeros(k, numOfFeatures);
    clusterSizes = zeros(k, 1);
    for rowIx = 1:numOfRows
        minDist = realmax;
        assignTo = 0;
        for centerIx = 1:k
            % Euclidian distance is used.
            %dist = sqrt(sum((X(rowIx, : ) - centers(centerIx, :)).^2));
            dist = custom_metric(X(rowIx,:),centers(centerIx,:));
            if dist < minDist
                minDist = dist;
                assignTo = centerIx;
            end
        end
        assignments(rowIx) = assignTo;

        % Keep these information to calculate cluster centers.
        clusterTotals(assignTo, :) = clusterTotals(assignTo, :) + X(rowIx, :);
        clusterSizes(assignTo) = clusterSizes(assignTo) + 1;
    end

    % This process is called 'singleton' in terms of Matlab.
    % If a cluster is empty choose a random data point as new
    % cluster cener.
    for clusterIx = 1:k
        if (clusterSizes(clusterIx) == 0)
            randomRow = round(1 + rand() * (numOfRows - 1) );
            clusterTotals(clusterIx, :) =  X(randomRow, :);
            clusterSizes(clusterIx) = 1;
        end
    end

    newCenters = zeros(k, numOfFeatures);
    for centerIx = 1:k
        newCenters(centerIx, :) = clusterTotals(centerIx, : ) / clusterSizes(centerIx);
    end

    diff = sum(sum(abs(newCenters - centers)));

    if diff < eps
        %disp('Centers are same, which means we converged before maxiteration count. This is a good thing!')
        break;
    end

    centers = newCenters;
end
assignments = assignments';
%printf('iter: %d, diff: %f\n', iter, diff);
end