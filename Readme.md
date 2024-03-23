# Neural Machine Translation with PyTorch

This small project was developed to gain hands-on experience with Neural Machine Translation (NMT) systems by building one from the ground up. It helped in understanding the intuition behind complex NMT systems. The model utilizes a bi-directional LSTM (biLSTM) and a cross-attention mechanism. During training, the teacher forcing strategy was adopted to enhance learning efficiency. The dataset comprises English to Portuguese translations and is located at `data/por-eng/por.txt`.

<p align="center">
    <img src="data/model.png" alt="architecture" width="40%">
</p>

## Results

Below are the evaluation scores:

| Metrics | Scores |
|---------|--------|
| BLEU-1 | 0.9639 |
| ROUGE-1 F-measure | 0.6756 |
| ROUGE-1 Precision | 0.6761 |
| ROUGE-1 Recall | 0.6898 |
| ROUGE-2 F-measure | 0.4788 |
| ROUGE-2 Precision |0.4798 |
| ROUGE-2 Recall | 0.4901 |
| ROUGE-L F-measure | 0.6676 |
| ROUGE-L Precision | 0.6682 |
| ROUGE-L Recall | 0.6816 |
| ROUGE-Lsum F-measure |0.6677 |
| ROUGE-Lsum Precision | 0.6682 |
| ROUGE-Lsum Recall | 0.6817 |

Here is a table showing translation results for a selection of sentences:

| English | Portuguese |
|---------|------------|
| She reads a book. | Ela lê um livro. |
| We enjoy the sunrise. | Nós gostamos do nascer do sol. |
| The cat sleeps on the sofa. | O gato dorme no sofá. |
| He cooks dinner. | Ele prepara o jantar. |
| Birds fly south in the winter. | Pássaros voam para o sul no inverno. |