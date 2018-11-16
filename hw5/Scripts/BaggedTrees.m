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
oobPredictions = zeros(N,numBags);
oobErrors = zeros(numBags,1);
for bag=1:numBags   
    % Find all the samples that don't appear in the bag
    xNotInBag = X(oobIndexesForBag{bag},:);
    % Get classifications for sample not in this bag
    oobPredictions(oobIndexesForBag{bag},bag) = predict(bags{bag}, xNotInBag);
    % Get majority vote
    oobMajority = sign(sum(oobPredictions(:,1:bag),2));
    oobMajority(oobMajority == 0) = randsample([-1 1],1);
    % Get the indexes for samples that we could get oob info for
    %oobIndexes = find(oobMajority);
    % Calculate average number of oob missclassifications
    %oobErrors(bag,1) = sum(oobMajority(oobIndexes,1) ~= Y(oobIndexes,1))/length(oobIndexes);
    oobErrors(bag,1) = sum(oobMajority ~= Y)/N;
end

plot(oobErrors)
oobErr = oobErrors(end);
end

function [OutOfBagError] = CalculateOutOfBagError(X, Y, bags, oobIndexesForBag)
    N = size(X,1);
    numBags = size(bags,1);
    
    classificationList = zeros(N,numBags);
    for bag=1:numBags
        % Find all the samples that don't appear in the bag
        xNotInBag = X(oobIndexesForBag{bag},:);
        
        % Get classifications for sample not in this bag
        classificationList(oobIndexesForBag{bag},bag) = predict(bags{bag}, xNotInBag);
    end
    
    % Get majority vote for classification across all samples
    oobClassAll = zeros(N,1);
    for i=1:size(classificationList,1)
        temp = classificationList(i,:);
        
        % Remove zeros
        temp(temp == 0) = [];
        if isempty(temp)
            oobClassAll(i,1) = 0;
        else
            [M,F,C] = mode(temp);
            if length(C) > 1
                % If there is a tie randomly select one of the values
                oobClassAll(i,1) = randsample(C,1);
            else
                oobClassAll(i,1) = M;
            end
        end 
    end
    
    % Get the indexes for samples that we could get oob info for
    oobClassIndexes = find(oobClassAll);
    oobClass = oobClassAll(oobClassIndexes,1);
    oobY = Y(oobClassIndexes,1);
    
    % Compare ensemble classification to real classification
    missclassCount = sum(oobClass ~= oobY);

    % Return average classification error
    OutOfBagError = missclassCount/size(oobClass,1);
end



