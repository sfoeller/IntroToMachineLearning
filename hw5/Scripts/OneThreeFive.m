% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)
load zip.train;
testData = load('zip.test');
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%
% One vs. Three Problem
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
m = TreeBagger(200,X,Y,'OOBPrediction','On';
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X, Y, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);
%%
% Get the test subsamples we care about
test_subsample = testData(find(testData(:,1)==1 | testData(:,1) == 3),:);
test_Y = test_subsample(:,1);
test_X = test_subsample(:,2:257);
test_N = size(test_X,1);

% Testing one decision tree
t = fitctree(X,Y); 
tPredictions = predict(t, test_X);
Eout = sum(tPredictions ~= test_Y)/test_N;
fprintf('The Error Out for the one-vs-three problem using 1 tree is: %.4f\n', Eout);

% Testing ensemble of 200 trees
[bags, oobIndexesForBag] = CreateBags(X, Y, 200);
[predictions] = GetEnsemblePredictions(test_X, bags);
Eout = sum(predictions ~= test_Y)/test_N;
fprintf('The Error Out for the one-vs-three problem using 200 trees is: %.4f\n', Eout);

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Three vs. Five Problem
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X, Y, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);

% Get the test subsamples we care about
test_subsample = testData(find(testData(:,1)==3 | testData(:,1) == 5),:);
test_Y = test_subsample(:,1);
test_X = test_subsample(:,2:257);
test_N = size(test_X,1);

% Testing one decision tree
t = fitctree(X,Y); 
tPredictions = predict(t, test_X);
Eout = sum(tPredictions ~= test_Y)/test_N;
fprintf('The Error Out for the three-vs-five problem using 1 tree is: %.4f\n', Eout);

% Testing ensemble of 200 trees
[bags, oobIndexesForBag] = CreateBags(X, Y, 200);
[predictions] = GetEnsemblePredictions(test_X, bags);
Eout = sum(predictions ~= test_Y)/test_N;
fprintf('The Error Out for the three-vs-five problem using 200 trees is: %.4f\n', Eout);

