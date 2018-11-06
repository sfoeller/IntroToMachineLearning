load expMedianOut.mat
load experimentOut.mat

Q_f = 5:5:20; % Degree of true function
N = 40:40:120; % Number of training examples
var = 0:0.5:2; % Variance of stochastic noise

% FIGURE 1
figure(1)
exp1 = expt_data_mat(4,1:end,1:end); % Get overfit measure for Qf = 20
exp1 = squeeze(exp1)'; % Drop the singleton dimension (Qf)
 
 %// Define integer grid of coordinates for the above data
[X,Y] = meshgrid(1:size(exp1,2), 1:size(exp1,1));
%// Define a finer grid of points
[X2,Y2] = meshgrid(1:0.01:size(exp1,2), 1:0.01:size(exp1,1));
%// Interpolate the data and show the output
outData = interp2(X, Y, exp1, X2, Y2, 'linear');

imagesc(outData);

%// Cosmetic changes for the axes
set(gca,'Ydir','Normal')
set(gca, 'XTick', linspace(1,size(X2,2),size(X,2))); 
set(gca, 'YTick', linspace(1,size(X2,1),size(X,1)));
set(gca, 'XTickLabel', N);
set(gca, 'YTickLabel', var);
xlabel('Number of Data Points, N');
ylabel('Noise Level, \sigma^{2}');
title({'Average Overfit Measure for';'Noise Vs. Number of Data Points'; 'Qf = 20'});

%// Add colour bar
c = colorbar;
colormap('jet');
c.Label.String = 'Overfit Measure'
caxis([-0.2 0.2])

% FIGURE 2
figure(2)
exp2 = expt_data_mat(1:end,1:end,3); % Get overfit measure for var = 0.1
exp2 = squeeze(exp2); % Drop the singleton dimension (Qf)
 
 %// Define integer grid of coordinates for the above data
[X,Y] = meshgrid(1:size(exp2,2), 1:size(exp2,1));
%// Define a finer grid of points
[X2,Y2] = meshgrid(1:0.01:size(exp2,2), 1:0.01:size(exp2,1));
%// Interpolate the data and show the output
outData = interp2(X, Y, exp2, X2, Y2, 'linear');

imagesc(outData);

%// Cosmetic changes for the axes
set(gca,'Ydir','Normal')
set(gca, 'XTick', linspace(1,size(X2,2),size(X,2))); 
set(gca, 'YTick', linspace(1,size(X2,1),size(X,1)));
set(gca, 'XTickLabel', N);
set(gca, 'YTickLabel', Q_f);
xlabel('Number of Data Points, N');
ylabel('Target Complexity, Qf');
title({'Average Overfit Measure for';'Qf Vs. Number of Data Points'; '\sigma^{2} = 0.1'});

%// Add colour bar
c = colorbar;
colormap('jet');
c.Label.String = 'Overfit Measure'
caxis([-0.2 0.2])

% FIGURE 3
figure(3)
exp3 = expt_data_median(4,1:end,1:end); % Get overfit measure for Qf = 20
exp3 = squeeze(exp3)'; % Drop the singleton dimension (Qf)
 
 %// Define integer grid of coordinates for the above data
[X,Y] = meshgrid(1:size(exp3,2), 1:size(exp3,1));
%// Define a finer grid of points
[X2,Y2] = meshgrid(1:0.01:size(exp3,2), 1:0.01:size(exp3,1));
%// Interpolate the data and show the output
outData = interp2(X, Y, exp3, X2, Y2, 'linear');

imagesc(outData);

%// Cosmetic changes for the axes
set(gca,'Ydir','Normal')
set(gca, 'XTick', linspace(1,size(X2,2),size(X,2))); 
set(gca, 'YTick', linspace(1,size(X2,1),size(X,1)));
set(gca, 'XTickLabel', N);
set(gca, 'YTickLabel', var);
xlabel('Number of Data Points, N');
ylabel('Noise Level, \sigma^{2}');
title({'Median Overfit Measure for';'Noise Vs. Number of Data Points'; 'Qf = 20'});

%// Add colour bar
c = colorbar;
colormap('jet');
c.Label.String = 'Overfit Measure'
caxis([-0.2 0.2])

% FIGURE 2
figure(4)
exp4 = expt_data_median(1:end,1:end,3); % Get overfit measure for var = 0.1
exp4 = squeeze(exp4); % Drop the singleton dimension (Qf)
 
 %// Define integer grid of coordinates for the above data
[X,Y] = meshgrid(1:size(exp4,2), 1:size(exp4,1));
%// Define a finer grid of points
[X2,Y2] = meshgrid(1:0.01:size(exp4,2), 1:0.01:size(exp4,1));
%// Interpolate the data and show the output
outData = interp2(X, Y, exp4, X2, Y2, 'linear');

imagesc(outData);

%// Cosmetic changes for the axes
set(gca,'Ydir','Normal')
set(gca, 'XTick', linspace(1,size(X2,2),size(X,2))); 
set(gca, 'YTick', linspace(1,size(X2,1),size(X,1)));
set(gca, 'XTickLabel', N);
set(gca, 'YTickLabel', Q_f);
xlabel('Number of Data Points, N');
ylabel('Target Complexity, Qf');
title({'Median Overfit Measure for';'Qf Vs. Number of Data Points'; '\sigma^{2} = 0.1'});

%// Add colour bar
c = colorbar;
colormap('jet');
c.Label.String = 'Overfit Measure'
caxis([-0.2 0.2])



