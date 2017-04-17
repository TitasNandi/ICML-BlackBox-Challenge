function [best cv_acc] = linearSearch(y, x, vals, type="linear", extra=5)
% Function searches a range of hyerparameter values for a linear kernel svm.
% It can be a c-svc or a nu-svc. 
%
% Params:
%    y - labels
%    x - data
%    vals - vector of values of c or nu to try
%    type - default "linear", type of svm to use. 
%     Options: "linear", "rbf", or "nu", which gives a linear kernel nu-svc
%    extra - the c value to pass to linearSearch if searching over g for the
%            rbf kernel
%
% Return:
%    best - the parameter value with the highest 10-fold cv accuracy
%    cv_acc - all of the 10-fold cv accuracies.

if(strcmp(type, "nu")),
  params = "-t 0 -s 1 -n %f -q -v 10";
elseif(strcmp(type, "linear"))
  params = "-t 0 -c %f -q -v 10";
elseif(strcmp(type, "rbf"))
  end_str = sprintf("-c %f -q -v 10", extra);
  params = ["-g %f ", end_str]; 
end
cv_acc = [];

for v = vals,
  param_str = sprintf(params, v);
  acc = svmtrain(y, x, param_str);
  cv_acc = [cv_acc acc];
end

[dummy idx] = max(cv_acc);
best = vals(idx);