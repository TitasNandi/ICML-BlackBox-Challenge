# utility script that takes the large csv file with the unlabeled data
# and chops it up into bite-sized pieces
# author: David Thaler

import csv

INFILE = '../data/unlabeled/extra_unsupervised_data.csv'
OUTFILE_STR = '../data/unlabeled/extra_%d_%d.csv'
BATCH_SZ = 5000
f_in = open(INFILE,'rb')
reader = csv.reader(f_in)
f_out = open(OUTFILE_STR % (1, BATCH_SZ),'wb')
writer = csv.writer(f_out)
idx = 0
for row in reader:
  idx += 1
  writer.writerow(row)
  if idx % BATCH_SZ == 0:
    f_out.close()
    end = idx + BATCH_SZ
    start = idx + 1
    f_out = open(OUTFILE_STR % (start, end),'wb')
    writer = csv.writer(f_out)

#NB: this leaves the end of the last file slightly misnamed/oddly-sized
f_out.close()