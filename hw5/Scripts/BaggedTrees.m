function [ oobErr ] = BaggedTrees( X, Y, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function   

[bags, oobIndexesForBag] = CreateBags(X, Y, numBags);

N = size(X,1);
oobErrors = NaN(numBags,1);
accumPred = NaN(N,2);
dataOobCount = zeros(N,1);

for bag=1:numBags
    % Get current tree
    tree = bags{bag};
    % Get this bags oob indexes
    oobI = oobIndexesForBag(:,bag);
    
    % If there are NaN rows in input scores, treat them as "no observations" and reset to 0
    nanscore = all(isnan(accumPred),2) & oobI;
    accumPred(nanscore,:) = 0;

    % Get data, weights and scores for this tree
    xNotInBag = X(oobI,:);
    
    % Get predictions for oob data.
    [~,~,nodes] = predict(tree,xNotInBag);
    
    % Get the probability for the classification
    predProb = tree.ClassProb(nodes,:);

    % Update how much this tree affects the prediction of each data point.
    tempOobCount = dataOobCount(oobI) + 1;

    % Get difference of old accumulation of predictions and the current trees prediction
    delta = predProb - accumPred(oobI,:);
    % We get this tree's predictions worth by dividing by how much its
    % contributing to the total prediction for each data point.
    gamma = bsxfun(@rdivide,delta,tempOobCount);
    % Update the accumulated predictions of the trees.
    accumPred(oobI,:) = accumPred(oobI,:) + gamma;

    % Save the current number of predictions for each data point.
    dataOobCount(oobI) = tempOobCount;

    % Find class with max probability for classification
    oobPredictions = zeros(N,1);
    
    % Find rows that we were able to find oob info for
    notNaN = ~all(isnan(accumPred),2); 

    % Find class with max prob
    [~,classNum] = max(accumPred(notNaN,:),[],2);
    oobPredictions(notNaN) = tree.ClassNames(classNum);

    oobSamplesCounted = length(oobPredictions(notNaN));
    if oobSamplesCounted > 0
        oobErrors(bag) = sum(oobPredictions(notNaN) ~=Y(notNaN))/oobSamplesCounted;
    end
end

plot(oobErrors)
oobErr = oobErrors(end);
end


