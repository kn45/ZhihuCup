

| Model | Description | Local score P/R/S | Submit score |
| ----- | ----------- | ----------------- | ------------ |
| ftxt1 | fasttext with title words | NA/NA/0.34 | 0.34 |
| ftxt2 | fasttext with doc words | 0.64/0.27/0.19 | NA |
| ftxt3 | bagging of ftxt1 & ftxt2 | 1.2/0.5/0.35 | 0.35 |
| ftxt4 | like ftxt1, use inst with single label to train | 0.89/0.36/0.26 | NA |
| cnn1  | cnn with title words, single label task | 1.09/0.45/0.32 | NA |
| rnn1  | lstm with title words | 1.23/0.50/0.36 | 0.36 |
| rnn2  | lstm with title words + doc words | 1.13/0.47/0.33 | 0.34 |
| ftxt5 | fasttext with title words + doc words | 1.14/0.48/0.34 | 0.34 |
| cnn2  | cnn with title words, multilabel task, pretrained emb, no grad clip | NA | NA |

