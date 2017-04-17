function sets = bunchOfWtsTest(feature_cts, databatch_ct, iter,write_features = true, r_dir='/home/titas/icml/data/icml_2013_black_box/R/test')
% Generate a set of weights from sparse filter training using the bulk unlabeled
% data for the Kaggle Blackbox challenge. If feature_cts has length k, then
% the output is a k x 2 cell array with features for the ith filter in sets{i,1}
% and the weights in {i, 2}. The unlabeled data has a 5K row batch pulled
% databatch_ct times and run for iter iterations of the sparse filter training
% algorithm.
%
% Params:
%  feature_cts - vector of feature counts to use for each feature set.
%       e.g. [100 100 100] trains three sets of 100 features each.
%  databatch_ct - the number of times to sample the data batches
%  iter - the number of iterations sparseFiltering runs on each data batch.
%  write_features - (default true) If true, write the features to
%        a directory for R to pick them up in.
%  r_dir - the path to where the files should be written for R to get them,
%
% Return:
%  sets - a length(feature_cts) x 2 cell array in which the rows hold the 
%         features, weights and svm model for each feature set. 
%         The features of the kth set are in sets{k, 1} & the weights are
%         in sets{k, 2}.

% path to where you want to put the output features so that R can find them

OUTPATH = [r_dir "/features%d.out"];
LABELPATH = '/home/titas/icml/data/icml_2013_black_box/ytrain.mat';

% These are start indices for the 135K unlabeled data
STARTS = 1:5000:130000;

n_batches = length(STARTS);
set_ct = length(feature_cts);
sets = {};

for k = 1:set_ct,
  feature_ct = feature_cts(k);
  batch_starts = STARTS(unidrnd(n_batches, databatch_ct, 1));
  blocks = [batch_starts(:) iter*ones(databatch_ct, 1)];
  fprintf("Learning features for set %d\n", k);
  fflush(stdout);
  [sets{k, 1} sets{k, 2}] = trainUnlabeledTest(feature_ct, blocks);
  if(write_features),
    outfile = sprintf(OUTPATH, k);
    features = sets{k,1};
    save(outfile, "features");
  end
end
