function [optW] = sparseFiltering(N, X, optW=[], max_iter=100)
% sparseFiltering trains a set of sparse filter features for some input
% 
% Params:
%   N - the number of features
%   X - the input data with examples in columns(the transpose of a design matrix).
%   optW - a set of input weights. If empty or not provided, 
%          they will be initialized randomly, which was the original behavior.
%   maxIter  - the iterations to run for
%
% Returns:
%   optW - a set of sparse filter features trained for (an additional)
%          maxIter iterations on X   
%
% NB: If optW is given, its row (output)dimension has to be N,
%     and its column dimension has to match the size of the input (rows of X).
%  
% Author: Jiquan Ngiam
% Modified: David Thaler
% Modifications:
%  1) Changed optimizer due to compatibility issues.If you have no issues with 
%     the mex files in minFunc, you should change it back.
%  2) This version can take weights and iteration counts as arguments.

    if(isempty(optW))
      % N = # features, X = input data (examples in column)
      optW = randn(N, size(X, 1));
    end
    
% If this package works for you, you should use it instead.
% Its included in the version of this code from Ngiam's repo.
%    optW = minFunc(@SparseFilteringObj, optW(:), ...
%                   struct('MaxIter', 200, 'Corr', 20), X, N);

% NB: fminunc (in Octave) runs out of memory on this input, so we don't use it.

% This is the old fmincg package from the Ng ML class
    options = optimset('MaxIter', max_iter);
    obj_fn = @(w)(SparseFilteringObj(w,X,N));
    optW = fmincg(obj_fn, optW(:),options);
    optW = reshape(optW, [N, size(X, 1)]);
    
end

function [Obj, DeltaW] = SparseFilteringObj (W, X, N)
    % Reshape W into matrix form
    W = reshape(W, [N, size(X,1)]);
    
    % Feed Forward
    F = W*X;
    Fs = sqrt(F.^2 + 1e-8);
    [NFs, L2Fs] = l2row(Fs);
    [Fhat, L2Fn] = l2row(NFs');
    
    % Compute Objective Function
    Obj = sum(sum(Fhat, 2), 1);
    
    % Backprop through each feedforward step
    DeltaW = l2rowg(NFs', Fhat, L2Fn, ones(size(Fhat)));
    DeltaW = l2rowg(Fs, NFs, L2Fs, DeltaW');
    DeltaW = (DeltaW .* (F./Fs)) * X';
    DeltaW = DeltaW(:);
end
