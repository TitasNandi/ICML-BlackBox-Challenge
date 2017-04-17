function model = modelFromWts(y, xtrain, wts, range, type="linear", extra=5)
% Function modelFromWts takes a set of weights for sparse filter features that
% are to be used whole, as-is. This is what you would use to run the provided
% weights.
%
% Params:
%   y - the labels
%   xtrain - the training set, as a design matrix
%   wts - weight matrix to give to feedForwardSF
%   range- vector of values of c or nu to try
%   type - default "linear", type of svm to use. 
%     Options: "linear", "rbf", or "nu", which gives a linear kernel nu-svc
%   extra - the c value to pass to linearSearch if searching over g for the
%           rbf kernel
%
% Return:
%   model - the model trained using those weights at the hyperparameter value
%           chosen in linearSearch.

f = feedForwardSF(wts, xtrain')';
best = linearSearch(y, f, range, type, extra);
if(strcmp(type, "nu")),
  params = sprintf("-t 0 -s 1 -n %f -q", best);
elseif(strcmp(type, "linear"))
  params = sprintf("-t 0 -c %f -q", best);
elseif(strcmp(type, "rbf"))
  params = sprintf("-g %f -c %f -q", best, extra);
end
model = svmtrain(y, f, params);