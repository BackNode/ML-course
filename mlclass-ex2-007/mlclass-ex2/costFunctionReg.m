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
J = sum( log( sigmoid(X * theta) ) .* (-y) - log( 1-sigmoid(X*theta) ) .* (1-y) ) / m;
J = J + (sum(theta.^2)-theta(1,1)^2)*lambda/(2*m);
grad(1,1) = X(:,1)' * (sigmoid(X*theta) - y) / m;
for j = 2:size(X,2),
    grad(j,1) = X(:,j)' * (sigmoid(X*theta) - y) / m + theta(j,1)*lambda/m;
end





% =============================================================

end
