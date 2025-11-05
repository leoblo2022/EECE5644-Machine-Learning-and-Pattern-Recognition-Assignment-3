%% Homework 3: Problem 2
%% Gaussian Mixture Models and K-fold Cross Validation for Model Order Selection

clear; clc; close all;

rng(42);

%% Step 1: Define Data Distributions
% Means - two of them close for overlap
%mu = [0 0; 3 3; 0 4; 4 0];
mu = [0 0; 2.5 3; 0 4; 6 0];

% Covariances - different covariance structures
Sigma(:,:,1) = [1 0.5; 0.5 1];
Sigma(:,:,2) = [1 -0.3; -0.3 1];
Sigma(:,:,3) = [0.5 0; 0 0.5];
Sigma(:,:,4) = [1.5 0; 0 1];

%Different Prior probabilities
priors = [0.2 0.3 0.28 0.22];

%% Step 2: Generate Data of different sizes
Ntrain = [10, 100, 1000];
iterations = 100; % repeat experiments

% Model orders to test
orderValues = 1:10;
numK = numel(orderValues);
K = 10;

% Store results
selection_counts = zeros(length(Ntrain), numK);

% Create GMM object
trueGMM = gmdistribution(mu, Sigma, priors);


%% Step 3 & 4: Run experiments
for i = 1:length(Ntrain)
    N = Ntrain(i);
    fprintf('Dataset size: %d\n', N);

    % Generate synthetic data from the true GMM
    X = random(gmdistribution(mu, Sigma, priors), N);

   % Visualize Data (overlay data samples with true distribution)
    figure;
    scatter(X(:,1), X(:,2), 30, 'filled');
    hold on;
    % Overlay contour of true GMM density
    x1 = linspace(-4, 8, 100);
    x2 = linspace(-4, 8, 100);
    [X1, X2] = meshgrid(x1, x2);
    Xgrid = [X1(:) X2(:)];
    pdfVals = pdf(trueGMM, Xgrid);
    contour(X1, X2, reshape(pdfVals, size(X1)), 10);
    title(sprintf('GMM Samples (N = %d)', Ntrain(i)));
    xlabel('x_1');
    ylabel('x_2');
    axis equal
    hold off;


    for rep = 1:iterations
        
        fprintf('Dataset size = %d | Repetition = %d / %d\n', N, rep, iterations);
        
        % Generate synthetic data from the true GMM
        X = random(gmdistribution(mu, Sigma, priors), N);
       
        % Divide the data set into K approximately-equal-sized partitions
        dummy = ceil(linspace(0,N,K+1));
        for k = 1:K
            indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
        end
        avgLogL = zeros(1, numK);
        
        % Evaluate candidate models with G components
        for ki = 1:numK
            G = orderValues(ki);
            logL_fold = zeros(K,1);

            for k = 1:K       
                % Manual 10-fold split indices
                indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
                Xtest = X(indValidate, :); % Using fold k as validation set
                %yValidate = y(indValidate);;
                if k == 1
                    indTrain = [indPartitionLimits(k+1,1):N];
                elseif k == K
                    indTrain = [1:indPartitionLimits(k-1,2)];
                else
                    indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
                end
                Xtrain = X(indTrain, :); % using all other folds as training set'
                %yTrain = y(indTrain);
                   
                try
                    % Fit GMM to training data
                    gm = fitgmdist(Xtrain, G, 'Options', statset('MaxIter', 1000, 'Display', 'off'), ...
                                   'RegularizationValue', 1e-6, 'Replicates', 3);

                    % Compute log-likelihood on test data
                    logL_fold(k) = sum(log(pdf(gm, Xtest)));
                catch
                    % In case of fitting failure, assign very low likelihood
                    logL_fold(k) = -inf;
                end
            end
            
            %Average CV log-likelihood
            avgLogL(ki) = mean(logL_fold);
        end
        
         % Select K with max average log-likelihood
        [~, bestK_idx] = max(avgLogL);
        bestK = orderValues(bestK_idx);
        selection_counts(i, bestK_idx) = selection_counts(i, bestK_idx) + 1;
    end
end

%% Step 5: Summarize and plot results
for i = 1:length(Ntrain)
    figure;
    bar(orderValues, selection_counts(i,:) / iterations);
    xlabel('Model Order (K)');
    ylabel('Selection Frequency');
    title(sprintf('Model Order Selection Frequency (N=%d)', Ntrain(i)));
    ylim([0 1]);
    grid on;
end

%% Display summary table
fprintf('\nSelection Frequencies (Proportion of times each K was chosen):\n');
for ni = 1:length(Ntrain)
    fprintf('\nN = %d\n', Ntrain(i));
    disp(array2table(selection_counts(i,:)/iterations, ...
        'VariableNames', compose('K%d', orderValues)));
end






