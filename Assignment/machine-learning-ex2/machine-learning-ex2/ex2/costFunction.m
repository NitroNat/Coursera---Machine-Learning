function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

predict_prob = sigmoid(X*theta);
% if predic_prob >= 0.5
%     predict = 1;
% else
%     predict = 0;
% end
% clever summing
J = (1/m)*( ((-y.*(log(predict_prob)))-(1-y).*log(1-predict_prob))' * ones(m,1) );

grad = (1/m)*( (predict_prob-y)' * X );







% =============================================================

end
