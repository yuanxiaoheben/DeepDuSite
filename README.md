# DeepDuSite
Method for dU sites prediction
### Pretrained
Using [BERT-pytorch](https://github.com/codertimo/BERT-pytorch) for pretrained, pretrained sequence data could download from [UCSC](http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz). The please transfer coodinate to DNA sequence, and then segment them to the corpus of one line with two 400bp adjacent DNA sequences.

### Train
Using [bert_ensemble_train.py](https://github.com/yuanxiaoheben/DeepDuSite/blob/master/bert_pytorch/bert_ensemble_train.py) for training.
 ``` shell
 python bert_ensemble_train.py \
 --train_corpus train_label.csv \
 --validation_corpus valid_label.csv \
 --bert_path bert.model.ep9 \
 --saved_path ./
 ```
