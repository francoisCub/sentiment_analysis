# Requires that IMDB is in .txt format (space between words, 1 sentence by line)
python models/wme.py IMDB_train.txt --R 300 --exp_id exp_train_300
python models/wme.py IMDB_test.txt --R 300 --exp_id exp_test_300