all:
	python 1_fit_models.py -o all -m ridgecv
	python 1_fit_models.py -o all -m ridgecv -s
other:
	python 1_fit_models.py -o cognitive -m ridgecv
	python 1_fit_models.py -o cognitive -m ridgecv -s
	python 1_fit_models.py -o perceptual-motor -m ridgecv
	python 1_fit_models.py -o perceptual-motor  -m ridgecv -s

clean:
	-rm /Users/poldrack/data_unsynced/multitask/encoding_models/*