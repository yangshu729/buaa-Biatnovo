#python Biatnovo/predict.py --model ~/experiment/biatnovo/translate.ckpt --predict_dir ~/experiment/biatnovo/predict2/ --predict_spectrum ~/data/train-data/ftp.peptideatlas.org/biatNovo/training.spectrum.mgf --predict_feature ~/data/train-data/ftp.peptideatlas.org/biatNovo/test_dataset_unique.csv --cuda
python Biatnovo/predict.py --model ~/experiment/biatnovo/translate.ckpt --predict_dir ~/experiment/biatnovo/predict2/ --predict_spectrum  ~/data/testing_uti.spectrum.mgf --predict_feature ~/data/testing_uti.feature.csv --cuda