%function bin_batch()
% A data-wrangling script that takes the 5000 row batches of unlabeled data in csv
% files and writes it as Octave binary for faster loading.
% author: David Thaler
DATAPATH = '/home/titas/icml/data/';
INFILE_STR = [DATAPATH 'icml_2013_black_box/csv/extra_%d_%d.csv'];
OUTFILE_STR = [DATAPATH 'icml_2013_black_box/csv/extra_%d_%d.mat'];

% NB: I have already done the first one, which has an odd first row.
%  This takes us from 5001-10000 through 135000-140000.
%  That last file is small (a few 100 rows).

BATCH_SZ = 5000;
END = 135000;
for k = 0:BATCH_SZ:END,
  start = k + 1;
  stop = k + BATCH_SZ;
  infile = sprintf(INFILE_STR, start, stop);
  x = csvread(infile);
  outfile = sprintf(OUTFILE_STR, start, stop);
  save("-binary", outfile, "x");
end
