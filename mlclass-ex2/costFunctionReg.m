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

n = size(theta);


errors = zeros(m,1);
for i=1:m,
  errors(i) = ( -y(i)*log(sigmoid(theta'*X(i,:)')) - (1-y(i))*log(1-sigmoid(theta'*X(i,:)')) );
end;
J = (sum(errors) / m) + (lambda * sum(theta(2:n).^2)) / (2*m);


%grad for theta0:
  inner = zeros(m,1);
  for i=1:m
    inner(i) = ( sigmoid(theta'*X(i,:)') - y(i) ) * X(i,1);
  end;
  grad(1) = sum(inner) / m;

%grad for other thetas:
for parameters=2:n
  inner = zeros(m,1);
  for i=1:m
    inner(i) = ( sigmoid(theta'*X(i,:)') - y(i) ) * X(i,parameters);
  end;
  grad(parameters) = (sum(inner) / m) + (lambda*theta(parameters))/m;
end;


% =============================================================

end
