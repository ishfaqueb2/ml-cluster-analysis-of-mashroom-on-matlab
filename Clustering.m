%% === Mushroom Clustering Analysis ===
% Dataset: UCI Mushroom Dataset (agaricus-lepiota.data)
% guidelines followed: PCA, K-means, Hierarchical, GMM, Silhouette

clear; clc; close all;

%% 1. Load Dataset
filename = 'agaricus-lepiota.data'; % make sure this file is in your MATLAB folder

% Force MATLAB to read it as text with comma delimiter
opts = detectImportOptions(filename, 'FileType', 'text', 'Delimiter', ',');
opts = setvartype(opts, 'string'); % read all columns as string

% Read dataset
dataRaw = readtable(filename, opts);

% First column = class (e = edible, p = poisonous) → Ignore for clustering
labels = dataRaw{:,1}; % keep for evaluation later
dataRaw(:,1) = [];


%% 2. Convert Categorical Data to Numeric
dataCat = table2array(dataRaw);
dataCat = categorical(dataCat);
dataNum = zeros(size(dataCat));

for j = 1:size(dataCat,2)
    [~,~,ic] = unique(dataCat(:,j)); % convert to numeric codes
    dataNum(:,j) = ic;
end

% Remove constant columns (no variability)
constantCols = std(dataNum) == 0;
dataNum(:, constantCols) = [];
fprintf('Removed %d constant columns.\n', sum(constantCols));

%% 3. PCA for Dimensionality Reduction
[pcs, scrs, ~, ~, pexp] = pca(dataNum);

% Show variance explained
figure;
pareto(pexp);
title('Variance Explained by Principal Components');

% Scatter plot of first 2 PCs
figure;
gscatter(scrs(:,1), scrs(:,2), labels, 'rb', 'ox');
xlabel('PC1'); ylabel('PC2');
title('PCA of Mushroom Dataset (True Labels Shown)');

%% 4. K-means Clustering
k = 2; % try 2 clusters (edible/poisonous)
[idx, C] = kmeans(dataNum, k, 'Replicates', 5, 'Distance', 'sqeuclidean');

% Silhouette plot
figure;
silhouette(dataNum, idx);
title('K-means Clustering Silhouette');

% PCA visualization of clusters
figure;
gscatter(scrs(:,1), scrs(:,2), idx);
xlabel('PC1'); ylabel('PC2');
title('K-means Clusters in PCA Space');

%% 5. Parallel Coordinates Visualization
figure;
parallelcoords(dataNum(:,1:6), 'Group', idx); % show first 6 attributes
title('Parallel Coordinates (first 6 attributes)');

%% 6. Hierarchical Clustering
Z = linkage(dataNum, "ward", "euclidean");
figure;
dendrogram(Z, 50); % show first 50 samples
title('Hierarchical Clustering Dendrogram');

% Assign groups
grp = cluster(Z, "maxclust", 2);

% PCA visualization
figure;
gscatter(scrs(:,1), scrs(:,2), grp);
xlabel('PC1'); ylabel('PC2');
title('Hierarchical Clustering in PCA Space');

%% 7. Gaussian Mixture Models (GMM)
gm = fitgmdist(dataNum, 2, 'RegularizationValue', 1e-6);
g = cluster(gm, dataNum);

figure;
gscatter(scrs(:,1), scrs(:,2), g);
xlabel('PC1'); ylabel('PC2');
title('GMM Clustering in PCA Space');


%% 8. Cluster Evaluation
clustev = evalclusters(dataNum, "kmeans", "silhouette", 'KList', 2:5);
disp(clustev);
fprintf('Best number of clusters (based on silhouette): %d\n', clustev.OptimalK);

%% 9. Compare Clusters with True Labels
% Convert true labels (e/p) to numeric
trueLabels = grp2idx(labels); % 1=edible, 2=poisonous

% Adjust clustering labels (because k-means may swap cluster IDs)
if mean(trueLabels(idx==1)) > mean(trueLabels(idx==2))
    idx = 3 - idx; % flip labels (1↔2)
end

% Confusion matrix
figure;
confusionchart(trueLabels, idx);
title('Confusion Matrix: K-means vs True Labels');

% Accuracy
acc = sum(trueLabels == idx) / length(trueLabels);
fprintf('Clustering accuracy vs ground truth: %.2f%%\n', acc*100);
% Hierarchical accuracy
grpLabels = grp;
if mean(trueLabels(grpLabels==1)) > mean(trueLabels(grpLabels==2))
    grpLabels = 3 - grpLabels;
end
acc_hier = sum(trueLabels == grpLabels) / length(trueLabels);
fprintf('Hierarchical accuracy: %.2f%%\n', acc_hier*100);

% GMM accuracy
gLabels = g;
if mean(trueLabels(gLabels==1)) > mean(trueLabels(gLabels==2))
    gLabels = 3 - gLabels;
end
acc_gmm = sum(trueLabels == gLabels) / length(trueLabels);
fprintf('GMM accuracy: %.2f%%\n', acc_gmm*100);
