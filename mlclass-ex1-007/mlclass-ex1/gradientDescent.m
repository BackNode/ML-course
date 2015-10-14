function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    J_dao1 = 0;
    J_dao2 = 0;
    for i = 1:m,
        J_dao1 = J_dao1 + ( X(i,:) * theta - y(i,:) ) * X(i,1);
        J_dao2 = J_dao2 + ( X(i,:) * theta - y(i,:) ) * X(i,2);
    end
    theta(1,1) = theta(1,1) - alpha * J_dao1 / m;
    theta(2,1) = theta(2,1) - alpha * J_dao2 / m;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
