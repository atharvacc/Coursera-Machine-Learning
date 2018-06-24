function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = size(X,2);
temp = X * theta;
h  = sigmoid(temp);
lhs =log(h);
rhs = (1 - (h));
rhs = log(rhs);
J = ((-1.*y).*lhs-((1.-y).*rhs));
J = (1/m) .* J;
J=sum(J);

lambda_sum = lambda/(2*m);
innersum=0;
for i=2:n,
innersum = innersum + ((theta(i,1))^2);
end;

r = lambda_sum * innersum;

J = J +r;

grad = (1/m) * X'*(h-y);
val2 = lambda/m;
temp3 = val2 .* theta;
temp3(1,1) = 0;
grad = grad + temp3



% =============================================================

end
