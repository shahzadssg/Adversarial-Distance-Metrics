% variation of DBSCAN with a custom metric "custom_metric"
% added comments to relate this code to the Pseudocode as given on https://en.wikipedia.org/wiki/DBSCAN

function [assignments,C] = dbscan_customDist(X,minpts,EPS,custom_metric)
C = 0;
assignments = zeros(size(X,1),1);
clustered = zeros(size(X,1),1);
for i=1: size(X,1)
    if(clustered(i)==1)
        continue;
    end
    clustered(i)=1;    %% mark P as visited
    isneighbour = [];
    neighbourcount = 0;
    for j=1: size(X,1)  %% N = D.regionQuery(P, eps)  "isneighbor" = N here
        %dist = sqrt(sum((X(i,:)-X(j,:)).^2));
        dist = custom_metric(X(i,:),X(j,:));
        if(dist<EPS)
            neighbourcount = neighbourcount + 1;
            isneighbour = [isneighbour j];
        end
    end
    if(neighbourcount<minpts)   %% if sizeof(N) < MinPts
        continue;  %% mark P as NOISE (not explicitly done)
    else
        C = C + 1;   %% C = next cluster; now "expandCluster(P, N, C, eps, MinPts)" begins
        assignments(i) = C;  %% add P to cluster C
        for k=isneighbour  %% for each point P' in N
            if(clustered(k)==0)  %% if P' is not visited
                clustered(k) = 1;  %% mark P' as visited
                isneighbour = [];
                neighbourcount = 0;
                for j=1: size(X,1)  %% N' = D.regionQuery(P', eps); "_isneighbour" = N' here
                    %dist = sqrt(sum((X(k,:)-X(j,:)).^2));
                    dist = custom_metric(X(k,:),X(j,:));
                    if(dist<EPS)
                        neighbourcount = neighbourcount + 1;
                        isneighbour = [isneighbour j];
                    end
                end
                if(neighbourcount>=minpts)  %% if sizeof(N') >= MinPts
                    isneighbour = [isneighbour isneighbour];  %% N = N joined with N'
                end
            end
            %% the next line from the pseudocode is missing in this implementation
            %%if P' is not yet member of any cluster
            assignments(k) = C;  %% add P' to cluster C
            %% unmark P' as NOISE if necessary => not necessary, since we never mark as noise
        end
    end
end