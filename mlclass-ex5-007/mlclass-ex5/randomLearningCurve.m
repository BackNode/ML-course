function [ error_train,error_val ] = ...
    randomLearningCurve( X, y, Xval, yval, lambda,random_times )
%RANDOMLEARNINGCURVE Summary of this function goes here
%   Detailed explanation goes here

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

theta = zeros(size(X, 2), 1); 
for i = 1:m
    for j = 1:random_times
        rand_train_vec = randperm(m,i);
        rand_val_vec = randperm( size(Xval,1),i);
        new_X = X(rand_train_vec,:);
        new_y = y(rand_train_vec,:);
        new_Xval = Xval(rand_val_vec,:);
        new_yval = yval(rand_val_vec,:);
        theta = trainLinearReg(new_X,new_y,lambda);
        error_train(i) = error_train(i) + linearRegCostFunction(new_X,new_y,theta,0);
        error_val(i) = error_val(i) + linearRegCostFunction(new_Xval,new_yval,theta,0);
    end
    error_train(i) = error_train(i) / random_times;
    error_val(i) = error_val(i) / random_times;
end

end

