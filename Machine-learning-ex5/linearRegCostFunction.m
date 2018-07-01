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

h = X * theta;
lhs = h -y;
temp= lhs;
lhs = lhs .^2;
lhs = sum(lhs);
lhs = (1/(2*m)) *lhs;

rhs = theta;
rhs = rhs .^2;
rhs(1,:) = 0;
rhs = sum(rhs);
rhs = (lambda/(2*m))*rhs;
% Costfunction
J = lhs +rhs;

%grad

lhs1 = X' * temp;
lhs1 = (1/m) .* lhs1;

rhs1 = theta;
rhs1(1,:) = 0;
rhs1 = (lambda/m) .* rhs1;

grad = lhs1 + rhs1;













% =========================================================================

grad = grad(:);

end
