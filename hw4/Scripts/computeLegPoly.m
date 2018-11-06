function [ z ] = computeLegPoly( x, Q )
%COMPUTELEGPOLY Return the Qth order Legendre polynomial of x
%   Inputs:
%       x: vector (or scalar) of reals in [-1, 1]
%       Q: order of the Legendre polynomial to compute
%   Output:
%       z: matrix where each column is the Legendre polynomials of order 0 
%          to Q, evaluated at the corresponding x value in the input

N = size(x,1);
z = zeros(Q+1,N);

z(1,1:N) = 1; % L0 = 1
z(2,1:N) = x; % L1 = x

for k=2:Q
    z(k+1,1:end) = ((2*k-1)/k)*x'.*z(k,1:end)-((k-1)/k)*z(k-1,1:end);
end
