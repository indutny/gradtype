GRADTYPE_RUN ?=

train: dataset
	python3 ./keras/train.py

tf-train: dataset
	python3 ./tf/train.py

tf-proxy: dataset
	python3 ./tf/train-proxy.py

tf-regression: dataset
	python3 ./tf/train-regression.py

tf-auto: dataset
	python3 ./tf/train-auto.py

regression: dataset
	python3 ./keras/train-regression.py

skipgrams: dataset
	python3 ./keras/train-skipgrams.py

dataset:
	npx ts-node ./bin/dataset.ts

clean:
	rm -rf out/*.raw
	rm -rf out/*.h5
	rm -rf logs/
	rm -rf images/

MODEL_WEIGHTS=$(wildcard saves/$(GRADTYPE_RUN)/*.index)
PCA_IMAGES_PRE=$(subst saves/$(GRADTYPE_RUN)/,images/pca/$(GRADTYPE_RUN)-, \
							 $(MODEL_WEIGHTS))
PCA_IMAGES=$(subst .index,.png, $(PCA_IMAGES_PRE))

images/pca.mp4: visualize
	ffmpeg -v quiet -y -r 10 -pattern_type glob \
		-i "images/pca/$(GRADTYPE_RUN)-*.png" \
		-vcodec libx264 -preset veryslow -pix_fmt yuv420p $@

visualize: images/pca $(PCA_IMAGES)
	echo $(MODEL_WEIGHTS)

images/pca:
	mkdir -p images/pca

images/pca/$(GRADTYPE_RUN)-%.png: saves/$(GRADTYPE_RUN)/%.index
	python3 ./tf/visualize.py $< $@

.PHONY: train train-tf regression dataset clean visualize skipgrams
