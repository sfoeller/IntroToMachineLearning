function [predictions] = GetEnsemblePredictions(X, bags)
%GETENSEMBLEPREDICTION Summary of this function goes here
%   Detailed explanation goes here

    N = size(X,1);
    numBags = size(bags,1);
    
    classificationList = zeros(N,numBags);
    for bag=1:numBags        
        % Get classifications for sample not in this bag
        classificationList(:,bag) = predict(bags{bag}, X);
    end
    
    [M,F,C] = mode(classificationList, 2);
    predictions = cellfun(@HandleTies, C);
end

% Check if there is a tie and if there is randomly select one of the chocies
function [choice] = HandleTies(c)
    if length(c) > 1
        choice = randsample(c,1);
    else
        choice = c;
    end
end

