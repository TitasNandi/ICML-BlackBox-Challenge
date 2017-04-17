if __name__ == '__main__':
	f = open('submission.csv', 'w')
	with open('pred_labels_test.txt', 'r') as reader:
		f.write('Id,Class\n')
		count = 1
		for line in reader:
			label = int(line.strip()[0])+1
			f.write(str(count)+','+str(label)+'\n')
			count += 1

	f.close()

