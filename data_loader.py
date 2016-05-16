import codecs
from collection import defaultdict as dd

src_train_file = codecs.open(sys.argv[1],'r','utf-8')
tgt_train_file = codecs.open(sys.argv[2],'r','utf-8')
count_cutoff = int(sys.argv[3])

src_counts = dd(int)
tgt_counts = dd(int)
for line in src_train_file:
	line = line.replace('\n','').split(' ')
	for word in line:
		src_counts[word]+=1
for line in tgt_train_file:
	line = line.replace('\n','').split(' ')
	for word in line:
		tgt_counts[word]+=1
src_words = [x[1] for x in src_counts if src_counts[x] >= count_cutoff]
tgt_words = [x[1] for x in tgt_counts if tgt_counts[x] >= count_cutoff]
src_vocab_mapping = {}
tgt_vocab_mapping = {}
for idx,word in enumerate(src_words):
	src_vocab_mapping[idx] = word
for idx,word in enumerate(tgt_words):
	tgt_vocab_mapping[idx] = word

	
