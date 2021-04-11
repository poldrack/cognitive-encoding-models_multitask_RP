all: clean base other

base:
	python 1_fit_models.py -o all -m ridgecv
	python 1_fit_models.py -o all -m ridgecv -s
other:
	python 1_fit_models.py -o cognitive -m ridgecv
	python 1_fit_models.py -o perceptual-motor -m ridgecv

clean:
	-rm /Users/poldrack/data_unsynced/multitask/encoding_models/*