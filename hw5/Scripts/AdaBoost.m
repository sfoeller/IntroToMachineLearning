function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use

trainN = size(X_tr, 1);
testN = size(X_te, 1);
w = ones(trainN, 1)./trainN;

% Create Adaboost classifier
weakLearners = cell(trainN,2);
for i=1:n_trees
    % Fit a classifier to the training data using the observation weights.
    cb = fitctree(X_tr,y_tr,'weights',w,'SplitCriterion','deviance','MaxNumSplits',1);
    %view(cb,'mode','graph') % graphic description
    % Compute the weighted misclassificaiton error
    cbX = cb.predict(X_tr);
    missclass = (cbX ~= y_tr);
    %missCount = length(find(missclass));
    errb = (w'*missclass)/sum(w);
    %Compute alpha
    alpha = log((1-errb)/errb);
    % Update the weights
    w = w.*exp(alpha*missclass);
    % store the weakLearner
    weakLearners{i,1} = cb;
    weakLearners{i,2} = alpha;
end

% Calculate error for both the testing and training data set.
pTrain = zeros(trainN, n_trees);
pTest = zeros(testN, n_trees);
train_err = zeros(n_trees,1);
test_err = zeros(n_trees,1);
for tree=1:n_trees
    pTrain(:,tree) = weakLearners{tree,2}.*weakLearners{tree,1}.predict(X_tr);
    pTest(:,tree) = weakLearners{tree,2}.*weakLearners{tree,1}.predict(X_te);
    train_err(tree) = sum(sign(sum(pTrain(:,1:tree),2)) ~= y_tr)/trainN;
    test_err(tree) = sum(sign(sum(pTest(:,1:tree),2)) ~= y_te)/testN;
end


end

