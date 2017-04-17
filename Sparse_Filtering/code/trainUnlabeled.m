function [w] = trainUnlabeled(feature_ct, blocks)
% trainUnlabeled trains a weight matrix on the unlabeled data in the Kaggle
% Blackbox challenge. 
%
% Params:
%   feature_ct - the number of features to have in this filter
%   blocks - b x 2 matrix where each row is a block. 
%     Each block has a start index and an iteration count.
%
% Return:
%   f - the features for xtrain evaluated on these weights
%   w - the weights learned
DATAPATH = '/home/titas/icml/data/icml_2013_black_box/csv/extra_%d_%d.mat';
TRAINPATH = '/home/titas/icml/data/icml_2013_black_box/csv/xtrain.mat';
TESTPATH = '/home/titas/icml/data/icml_2013_black_box/csv/xtest.mat';
BATCH_SIZE = 5000;
blocks = blocks';
w = [];
for b = blocks,
  start = b(1);
  infile = sprintf(DATAPATH, start, start + BATCH_SIZE - 1);
  % NB: this leaves a 5000 x 1875 design matrix x in the local namespace
  load(infile);
  fprintf("Loading: %s\n", infile);
  fflush(stdout);
  w = sparseFiltering(feature_ct, x', w, b(2));
end

