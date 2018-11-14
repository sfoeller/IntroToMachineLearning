function [bags, oobIndexesForBag] = CreateBags(X, Y, numBags)
%CREATEBAGS Summary of this function goes here
%   Detailed explanation goes here
    bags = cell(numBags,1);
    oobIndexesForBag = cell(numBags,1);
    
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
        oobIndex = true(size(X,1),1);
        uniqueI = unique(sampleIndexes);
        oobIndex(uniqueI,1) = false;
        
        % Store indexes of samples that aren't in this bag
        oobIndexesForBag{i} = oobIndex;
    end
end

