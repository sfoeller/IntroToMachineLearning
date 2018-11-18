function [bags, oobIndexesForBag] = CreateBags(X, Y, numBags)
%CREATEBAGS Summary of this function goes here
%   Detailed explanation goes here
    N = size(X,1);
    bags = cell(numBags,1);
    oobIndexesForBag = false(N,numBags);
    
    % Put our data samples and class back together for sampling below
    data = [X Y]; 
    
    for i=1:numBags
        % Get bootstrap sample
        [bootstrapSample, sampleIndexes] = datasample(data,size(data,1));
        
        % Seperate data and classifications 
        X = bootstrapSample(:,1:end-1);
        Y = bootstrapSample(:,end); 
        
        % Create tree using bootstrap sample
        bags{i} = fitctree(X, Y);
        
        % Record indices of OOB instances
        oobIndex = true(N,1);
        oobIndex(sampleIndexes) = false;
        oobIndexesForBag(:,i) = oobIndex;
    end
end

