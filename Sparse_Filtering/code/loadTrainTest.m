function [f] = loadTrainTest(w, path)

load(path);

% Make features for these weights
f = feedForwardSF(w, x');
f = f';
