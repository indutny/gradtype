build:
	./bin/dataset.ts datasets/verify/verify-both.json > datasets/verify/verify-both.csv
	./bin/dataset.ts datasets/verify/verify-left.json > datasets/verify/verify-left.csv
	./bin/dataset.ts datasets/verify/verify-right.json > datasets/verify/verify-right.csv
	./bin/combine.ts datasets/verify/verify-both.csv datasets/verify/verify-left.csv datasets/verify/verify-right.csv > datasets/verify/combined.csv
	./bin/dataset.ts datasets/train/train-both.json > datasets/train/train-both.csv
	./bin/dataset.ts datasets/train/train-left.json > datasets/train/train-left.csv
	./bin/dataset.ts datasets/train/train-right.json > datasets/train/train-right.csv
	./bin/combine.ts datasets/train/train-both.csv datasets/train/train-left.csv datasets/train/train-right.csv > datasets/train/combined.csv

.PHONY: build
