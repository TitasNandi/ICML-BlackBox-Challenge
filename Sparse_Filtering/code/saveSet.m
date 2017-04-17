function saveSet(sets, r_dir='../r_work')
% Takes an already existing set of features and weights from bunchOfWts
% and saves its features at OUTPATH, where R can find them.
%
% Param:
%   sets - a set of features and weights from bunchOfWts 
%   r_dir  - the path to where the R functions will write their files.
%
% Return - none, but saves the features from the set at the location OUTPATH
%          where R will access them.

% path to where you want to put the output features so that R can find them
OUTPATH = [r_dir "/features%d.out"];
n = size(sets, 1);
for k=1:n
  outfile = sprintf(OUTPATH, k);
  features = sets{k,1};
  save(outfile, "features");
end