function basic_submit(pred, filename)
% Basic script for preparing a submission to the Kaggle
% black box challenge in its '1.0\n' format.
%
% Params:
%    pred - the predictions
%    filename - the file is written at './output/<filename>.txt'
%         NB:That's without the '.txt' in the argument.

FMT = "%.1f\n";
pathname = sprintf("output/%s.txt", filename);
fid = fopen(pathname, "a");
for k=1:length(pred)
  fprintf(fid, FMT, pred(k));
end
fclose(fid);
