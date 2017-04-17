function [wts, model] = buildModel(y, xtrain, sets, fraction, ...
                              range, type="linear", extra=5, rank_dir="output")
% buildModel takes a set of weights from bunchOfWts, loads feature ranks from
% files written on the R side of the system (rankAll.R), and produces a set of 
% selected and aggregated weights, as well as a model using those weights, 
% run at the hyperparamter value chosen by linearSearch from out of a range.
%
% Params:
%   y - the labels
%   xtrain - the training set, as a design matrix
%   sets - the cell array of features and weights left over from bunchOfWts,
%          The cell array may be size n x 2 or n x 3. If appendRanks has been
%          run, it will be the latter, and the feature ranks will be sought in
%          the 3rd column of sets. Otherwise, the ranks need to be in files
%          at RANKPATH (ie R needs to have written them there).
%   fraction - a decimal fraction of features to retain, eg 0.5 will keep
%            the higher-scoring half of features.
%   range - the range of hyperparameter values to try in linear search
%   type - default "linear", type of svm to use. 
%     Options: "linear", "rbf", or "nu", which gives a linear kernel nu-svc
%   extra - the c value to pass to linearSearch if searching over g for the
%           rbf kernel
%   rank_dir - the directory where R wrote the rank vectors out
%
% Return:
%   wts - the aggregated weights from all of the sets
%   model - the model trained using those weights at the hyperparameter value
%           chosen in linearSearch.

% If the set does not have ranks appended, they need to be in files here
RANKPATH = [rank_dir "/ranks%d.out"];

n = size(sets, 1);
has_ranks = (size(sets, 2) == 3);
wts = [];
for k=1:n
  w_k = sets{k, 2};
  if(has_ranks),
    rank_k = sets{k, 3};
  else,
    rank_path = sprintf(RANKPATH, k);
    rank_k = load(rank_path);
  end
  num_features = round(size(w_k, 1) * fraction);
  wts = [wts; w_k(rank_k(1:num_features), : )];
end
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
end
