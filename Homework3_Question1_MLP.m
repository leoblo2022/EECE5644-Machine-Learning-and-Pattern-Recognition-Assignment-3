%% Homework 3, Question 1: Multilayer Perceptron (MLP)
clear all; close all;

%% Data Distribution

% Data paramaters
Ntrain_list = [100, 500, 1000, 5000, 10000];
Ntest  = 100000;   % test samples
C = 4; % number of classes
dim = 3; % dimension of data

priors = [.25 .25 .25 .25]; % uniform priors
% pick your own mean vectors and covariance matrices
mu = [ [4 4 0]; [3 3 0]; [0 3 3]; [3 0 2] ];
Sigma(:,:,1) = eye(3); 
Sigma(:,:,2) = [1 0.5 0; 0.5 1 0; 0 0 1]; 
Sigma(:,:,3) = [1 0 0.5; 0 1 0; 0.5 0 1]; 
Sigma(:,:,4) = eye(3);

% Visualize Data in 3D space
[xTrain, yTrain] = generateDataFromGMM(Ntrain_list(3), mu, Sigma, priors);    
figure;
scatter3(xTrain(1, yTrain==1), xTrain(2, yTrain==1), xTrain(3, yTrain==1), 20, 'r', 'filled'); hold on;
scatter3(xTrain(1, yTrain==2), xTrain(2, yTrain==2), xTrain(3, yTrain==2), 20, 'g', 'filled');
scatter3(xTrain(1, yTrain==3), xTrain(2, yTrain==3), xTrain(3, yTrain==3), 20, 'b', 'filled');
scatter3(xTrain(1, yTrain==4), xTrain(2, yTrain==4), xTrain(3, yTrain==4), 20, 'm', 'filled');
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
title('Generated Gaussian Data by Class');
legend('Class 1', 'Class 2', 'Class 3', 'Class 4');


%% MLP Structure
% mlp paramaters
inputDim = 3;     % number of input features (3D)
hiddenDim = 1;   % starting number of perceptrons, will be optimally selected later
outputDim = 4;    % number of classes
learningRate = 0.01;
numEpochs = 5000;

% create labels
Y = full(ind2vec(yTrain, outputDim));  % size: [outputDim x Ntrain]

% Initialize network weights
rng(0);
W1 = 0.1 * randn(hiddenDim, inputDim);
b1 = zeros(hiddenDim, 1);
W2 = 0.1 * randn(outputDim, hiddenDim);
b2 = zeros(outputDim, 1);

%% Generate Test Data
[xTest, yTest]   = generateDataFromGMM(Ntest, mu, Sigma, priors);
logLikelihood = zeros(outputDim, Ntest);

%% Theoretically Optimal Classifier
% Compute p(x | w_k) for each test point and class

for k = 1:outputDim
    % Compute log of pdf
    logLikelihood(k,:) = log(mvnpdf(xTest', mu(k,:), Sigma(:,:,k)) + 1e-300);
end

% Since priors are equal, posterior is proportional to likelihood
[~, yBayes] = max(logLikelihood, [], 1);

% Compute empirical probability of error
bayesAccuracy = mean(yBayes == yTest);
bayesError = 1 - bayesAccuracy;

fprintf('Theoretical Optimal Classifier');
fprintf('Bayes Accuracy: %.2f%%\n', bayesAccuracy * 100);
fprintf('Bayes Error: %.2f%%\n', bayesError * 100);
fprintf('\n')


%% Model Order Selection
Nfolds = 10; % 10-fold cross validation
perceptronList = [1, 5, 10, 15, 20];

% Loop over training set sizes
for s = 1:length(Ntrain_list)
    Ntrain = Ntrain_list(s);

    %% Generate Training Data 
    [xTrain, yTrain] = generateDataFromGMM(Ntrain, mu, Sigma, priors);

    % 10-fold cross-validation indices
    indices = makeFolds(length(yTrain), Nfolds); % used helper function

    meanValError = zeros(1, length(5));

    % Try several numbers-of-perceptrons
    for M = 1:length(perceptronList)
        numPerceptrons = perceptronList(M);
        valErrors = zeros(1, Nfolds);

        for fold = 1:Nfolds
            % Split train/validation sets
            valIdx = (indices == fold);
            trainIdx = ~valIdx;
            xTr = xTrain(:, trainIdx);
            yTr = yTrain(:, trainIdx);
            xVal = xTrain(:, valIdx);
            yVal = yTrain(:, valIdx);

            %% Model Training 
            [W1, b1, W2, b2] = trainMLP(xTr, yTr, inputDim, numPerceptrons, outputDim, learningRate, numEpochs); % used helper function

            % Evaluate on validation set
            yPredVal = predictMLP(xVal, W1, b1, W2, b2); % used helper function
            valErrors(fold) = mean(yPredVal ~= yVal);
        end 
        meanValError(M) = mean(valErrors);
        fprintf('N=%d, Hidden=%d, Mean Val Error=%.3f\n', Ntrain, numPerceptrons, meanValError(M));
    end 
    % Pick best model order
    [~, bestIdx] = min(meanValError);
    bestHidden(s) = perceptronList(bestIdx);

    % Retrain on full data with best hiddenDim
    [W1, b1, W2, b2] = trainMLP(xTrain, yTrain, inputDim, bestHidden(s), outputDim, learningRate, numEpochs);

    %% Performance Assessment 
    % Test on large test set
    yPred = predictMLP(xTest, W1, b1, W2, b2);
    accuracy = mean(yPred == yTest);
    testErrors(s) = mean(yPred ~= yTest);

    fprintf('=== Test Accuracy of mlp ===\n');
    fprintf('parameters: N=%d, Best number of perceptrons in hidden layer =%d, Test Error=%.3f, Test Accuracy: %.2f%%\n', Ntrain, bestHidden(s), testErrors(s)*100, accuracy * 100);
    fprintf('\n')
end 

%% Visualize Best Performance Results 
figure; hold on; 
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
title('MLP Classification Results');
view(45,25); axis equal;

% Define four distinct green shades
greenShades = [ 0.2 0.8 0.2; 0.0 0.6 0.0; 0.0 0.4 0.0; 0.3 0.7 0.3;];

% Loop over classes and plot correct/incorrect
for k = 1:outputDim
    correctIdx = find(yTest == k & yPred == k);
    incorrectIdx = find(yTest == k & yPred ~= k);

    % Correct: shaded green
    scatter3(xTest(1, correctIdx), xTest(2, correctIdx), xTest(3, correctIdx), ...
        20, greenShades(k,:), 'filled');

    % Incorrect: red
    scatter3(xTest(1, incorrectIdx), xTest(2, incorrectIdx), xTest(3, incorrectIdx), ...
        20, 'r', 'filled');
end
legend({'Correct (Class 1)', 'Correct (Class 2)', 'Correct (Class 3)', 'Correct (Class 4)', 'Incorrect'}, ...
    'Location', 'bestoutside');
axis equal;


%% Model Training function 
function [W1, b1, W2, b2] = trainMLP(xTrain, yTrain, inputDim, hiddenDim, outputDim, learningRate, numEpochs)
Y = full(ind2vec(yTrain, outputDim));
Ntrain = size(xTrain, 2);
rng(0);
W1 = 0.1 * randn(hiddenDim, inputDim);
b1 = zeros(hiddenDim, 1);
W2 = 0.1 * randn(outputDim, hiddenDim);
b2 = zeros(outputDim, 1);

for epoch = 1:numEpochs

    % Forward pass  step
    Z1 = W1 * xTrain + b1;      % hidden pre-activation
    A1 = activationFunction(Z1); % sigmoid activation helper function
    Z2 = W2 * A1 + b2;          % output pre-activation
    
    % Softmax output
    expZ = exp(Z2 - max(Z2,[],1)); % for numerical stability
    A2 = expZ ./ sum(expZ,1);
    
    % compute Loss (cross-entropy) 
    %loss = -mean(sum(Y .* log(A2 + 1e-12), 1));
    
    % Backpropagation step
    dZ2 = A2 - Y;                        % output layer gradient
    dW2 = (dZ2 * A1') / Ntrain;
    db2 = mean(dZ2, 2);
    
    dA1 = W2' * dZ2;
    dZ1 = dA1 .* A1 .* (1 - A1);         % derivative of sigmoid
    dW1 = (dZ1 * xTrain') / Ntrain;
    db1 = mean(dZ1, 2);
    
    % Gradient descent step
    W1 = W1 - learningRate * dW1;
    b1 = b1 - learningRate * db1;
    W2 = W2 - learningRate * dW2;
    b2 = b2 - learningRate * db2;
    
    % display progress
    %if mod(epoch, 10) == 0
        %fprintf('Epoch %d/%d, Loss = %.4f\n', epoch, numEpochs, loss);
    %end
end
end 

%% Performance Evaluation function
function yPred = predictMLP(x, W1, b1, W2, b2)
    Z1 = W1*x + b1; A1 = 1./(1+exp(-Z1));
    Z2 = W2*A1 + b2;
    expZ = exp(Z2 - max(Z2,[],1));
    A2 = expZ ./ sum(expZ,1);
    [~, yPred] = max(A2, [], 1);
end

%% Generate Data function
% helper function to generate data: Generates N samples from a C-class Gaussian mixture
function [x, labels] = generateDataFromGMM(N, mu, Sigma, priors)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = priors;
meanVectors = mu;
covMatrices = Sigma;

n = size(meanVectors,2);  % Data dimensionality
C = length(priors);    % Number of classes
x = zeros(n, N); labels = zeros(1, N);
Nl = floor(N / C);
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
idx = 1;
for l = 1:C
    Xk = mvnrnd(mu(l,:), Sigma(:,:,l), Nl)';
    x(:, idx:idx+Nl-1) = Xk;
    labels(idx:idx+Nl-1) = l;
    idx = idx + Nl;
end
end

%% k-fold cross validation function
function indices = makeFolds(N, K)
% makeFolds: manually create K roughly equal folds for N samples
% Returns an index vector of length N with fold numbers (1..K)

    indices = zeros(1, N);
    foldSizes = repmat(floor(N / K), 1, K);
    remainder = mod(N, K);
    foldSizes(1:remainder) = foldSizes(1:remainder) + 1;

    allIdx = randperm(N);  % randomize sample order
    startIdx = 1;

    for k = 1:K
        thisFold = allIdx(startIdx : startIdx + foldSizes(k) - 1);
        indices(thisFold) = k;
        startIdx = startIdx + foldSizes(k);
    end
end

% activation function
function out = activationFunction(in)
% Pick a shared nonlinearity for all perceptrons: sigmoid or ramp style...
% You can mix and match nonlinearities in the model.
% However, typically this is not done; identical nonlinearity functions
% are better suited for parallelization of the implementation.
out = 1./(1+exp(-in)); % Logistic function - sigmoid style nonlinearity
%out = in./sqrt(1+in.^2); % ISRU - sigmoid style nonlinearity
end

