%Script that runs the set of necessary experiments

Q_f = 5:5:20; % Degree of true function
N = 40:40:120; % Number of training examples
var = 0.5 % Variance of stochastic noise

% [p,q,r] = meshgrid(Q_f, N, var);
% pairs = [p(:) q(:) r(:)];
% numCombinations = size(pairs,1)

expt_data_mat = zeros(length(Q_f), length(N), length(var));

tic
for ii = 1:length(Q_f)
    for jj = 1:length(N)
        for kk = 1:length(var)
            expt_data_mat(ii,jj,kk) = mean(computeOverfitMeasure(Q_f(ii),N(jj),1000,var(kk),500));
        end
    end
    fprintf('.');
end
toc          