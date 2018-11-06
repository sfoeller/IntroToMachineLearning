function [ overfit_m ] = computeOverfitMeasure( true_Q_f, N_train, N_test, var, num_expts )
%COMPUTEOVERFITMEASURE Compute how much worse H_10 is compared with H_2 in
%terms of test error. Negative number means it's better.
%   Inputs
%       true_Q_f: order of the true hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       var: variance of the stochastic noise
%       num_expts: number of times to run the experiment
%   Output
%       overfit_m: vector of length num_expts, reporting each of the
%                  differences in error between H_10 and H_2

overfit_m = zeros(num_expts,1);

%tic;
for i=1:num_expts
    [train_set test_set] = generate_dataset(true_Q_f, N_train, N_test, sqrt(var));
    
    % Transform training set to Z-space
    g2_train = computeLegPoly(train_set(1:end, 1), 2); % Do the 2nd order legendre transform
    g10_train = computeLegPoly(train_set(1:end, 1), 10); % Do the 10th order legendre transform
    
    % Find optimal weight vector based on the training data in the Z-space
    g2_wlin = glmfit(g2_train', train_set(1:end,2), 'normal','constant','off'); % find 2nd order seperator
    g10_wlin = glmfit(g10_train', train_set(1:end,2), 'normal','constant','off'); % find 10th order seperator
    
    % Transform test set to Z-space
    g2_test = computeLegPoly(test_set(1:end, 1), 2);
    g10_test = computeLegPoly(test_set(1:end, 1), 10);
    
    % Get g2 and g10 using the seperators found above.
    g2_out = glmval(g2_wlin, g2_test','identity','constant','off');
    g10_out = glmval(g10_wlin, g10_test','identity','constant','off');
    
    % Calculate error for g2 and g10
    e2 = mean((g2_out -  test_set(1:end, 2)).^2);
    e10 = mean((g10_out -  test_set(1:end, 2)).^2);
    
    overfit_m(i) = e10 - e2;
end
%toc
