function [ train_set test_set ] = generate_dataset( Q_f, N_train, N_test, sigma )
%GENERATE_DATASET Generate training and test sets for the Legendre
%polynomials example
%   Inputs:
%       Q_f: order of the hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       sigma: standard deviation of the stochastic noise
%   Outputs:
%       train_set and test_set are both 2-column matrices in which each row
%       represents an (x,y) pair

train_set = zeros(N_train, 2);
test_set = zeros(N_test, 2);

% range of our x values
a = -1;
b = 1;

% Sample x values for training and test sets
train_set(1:end,1) = (b-a)*rand(N_train,1)+ a;
test_set(1:end,1) = (b-a)*rand(N_test,1)+ a;

% Generate coefficients with standard normal distribution
coeff = normrnd(0,1,Q_f+1,1); 

% Find value to scale coefficients by and apply it.
q = 0:Q_f;
coeff = coeff/sqrt(sum(1./(2*q+1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the y values for the training set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% noise follows a standard normal distribution, with our chosen sigma
noise = normrnd(0,sigma,1,N_train); 

% Real target function output
f = sum(coeff.*computeLegPoly(train_set(1:end, 1), Q_f));

% Noisy y output
train_set(1:end,2) = f+noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the y values for the test set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
noise = normrnd(0,sigma,1,N_test); 
f = sum(coeff.*computeLegPoly(test_set(1:end, 1), Q_f));
test_set(1:end,2) = f+noise;