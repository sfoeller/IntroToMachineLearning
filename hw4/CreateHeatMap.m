load experimentOut.mat

Q_f = 5:5:20; % Degree of true function
N = 40:40:120; % Number of training examples
var = 0:0.5:2; % Variance of stochastic noise
x = expt_data_mat(1,1:end,1:end);
x = squeeze(x)

 A = x;             % matrix to draw
 colormap('hot');   % set colormap
 imagesc(A);        % draw image and scale colormap to values range
 colorbar;          % show color scale

%h = heatmap(N,var,cdata)