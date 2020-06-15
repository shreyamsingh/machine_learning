function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hX = X * theta;
sqError = sum((hX - y) .^ 2);
regTerm = sum(theta(2:size(theta,1), :) .^ 2);
J = sqError/(2*m) + regTerm*(lambda/(2*m));

%grad(1) = (1/m) * sum((X(:, 1))' * (hX - y));
%grad(2:size(grad), :) = (1/m) * sum((X(:, 2:size(X, 2))')*(hX - y));
%grad = (1/m) * sum((X(:, 2:size(X, 2))')*(hX - y));
grad = ((1/m) * (hX - y)' * X)';
grad(2:size(grad)) = grad(2:size(grad)) + (lambda/m) * theta(2:size(theta), :);







% =========================================================================

grad = grad(:);

end
