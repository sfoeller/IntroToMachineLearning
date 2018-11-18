% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)

load ../zip.train;
testData = load('../zip.test');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% One vs. Three Problem
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);
% Cross Validation Error and OOB error.
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
figure(1);
clf(1);
bee = BaggedTrees(X, Y, 200);
title({'One-vs-Three Problem'; 'OOB Error vs. Number of Trees in Ensemble'});
xlabel('Number of Trees in Ensemble');
ylabel('Average OOB Error');
saveas(1,'OnevsThreeOobError.jpg');
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);

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
[bags, ~] = CreateBags(X, Y, 200);
[predictions] = GetEnsemblePredictions(test_X, bags);
Eout = sum(predictions ~= test_Y)/test_N;
fprintf('The Error Out for the one-vs-three problem using 200 trees is: %.4f\n', Eout);
%m = TreeBagger(200,X,Y,'OOBPrediction','On');
%m.DefaultYfit = '';
%e = error(m,test_X,test_Y);
%fprintf('The Error Out for the one-vs-three problem using 200 TreeBagertrees is: %.4f\n', e(end));
%fprintf('The OOB Error for the one-vs-three problem using 200 TreeBagertrees is: %.4f\n', oobError(m,'Mode','ensemble'));

% Testing adaboost
% Convert to positive and negative classifications
Y(Y == 1) = -1;
Y(Y == 3) = 1;
test_Y(test_Y == 1) = -1;
test_Y(test_Y == 3) = 1;

[ train_err, test_err ] = AdaBoost( X, Y, test_X, test_Y, 200);

figure(2)
clf(2)
hold on
plot(train_err);
plot(test_err);
title({'One-vs-Three classification with Adaboost';'Training and Test Error vs. # of weak hypotheses'});
xlabel('Number of weak learners');
ylabel('Average Error');
legend('Training Error', 'Test Error');
saveas(2,'OnevsThreeAdaBoostErr.jpg');
hold off

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Three vs. Five Problem
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y = subsample(:,1);
X = subsample(:,2:257);
% Cross Validation Error and OOB error.
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
figure(3);
clf(3);
bee = BaggedTrees(X, Y, 200);
title({'Three-vs-Five Problem'; 'OOB Error vs. Number of Trees in Ensemble'});
xlabel('Number of Trees in Ensemble');
ylabel('Average OOB Error');
saveas(3,'ThreevsFiveOobError.jpg');
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

% Testing adaboost
Y(Y == 3) = -1;
Y(Y == 5) = 1;
test_Y(test_Y == 3) = -1;
test_Y(test_Y == 5) = 1;
[ train_err, test_err ] = AdaBoost( X, Y, test_X, test_Y, 200);

figure(4)
clf(4)
hold on
plot(train_err);
plot(test_err);
title({'One-vs-Three classification with Adaboost';'Training and Test Error vs. # of weak hypotheses'});
xlabel('Number of weak learners');
ylabel('Average Error');
legend('Training Error', 'Test Error');
saveas(4,'ThreevsFiveAdaBoostErr.jpg');
hold off