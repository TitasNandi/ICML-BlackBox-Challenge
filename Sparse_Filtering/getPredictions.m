function pred = getPredictions(wts, model, xtest, submission_name)
% Utility function to bundle up the steps needed to turn weights and a model
% into a submission file in KaggleBlackbox,
%
% Params:
%   wts - the aggregated weights from all of the sets
%   model - the model trained using those weights at the hyperparameter value
%           chosen in linearSearch.
%   xtest - the test set, as a design matrix
%   submission_name - passed to basic_submit as the name of the submission file.
%
% Return:
%   pred - the vector of predictions

ftest = feedForwardSF(wts, xtest')';
dummy = ones(size(xtest, 1), 1);
pred = svmpredict(dummy, ftest, model);
basic_submit(pred, submission_name);
