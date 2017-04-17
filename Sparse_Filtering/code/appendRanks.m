function sets = appendRanks(sets, rank_dir="output")
% Utility script to take a set of ranks written by R and attach them to the end
% of the set of weights and features that they are for. If applied twice, the
% first ranks are overwritten.
%
% Param:
%   sets - a set of features and weights from bunchOfWts 
%   rank_dir - the directory where R wrote the rank vectors out
%
% Return - sets, with the ranks apended in the third column.

% path to where R wrote the ranks
RANKPATH = [rank_dir "/ranks%d.out"];
n = size(sets, 1);
for k=1:n
  rank_path = sprintf(RANKPATH, k);
  rank_k = load(rank_path);
  sets{k, 3} = rank_k;
end