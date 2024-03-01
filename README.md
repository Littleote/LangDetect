## Starting

This project is just a demo of a language detector.
It can be useful to play around with the effect of the different parameters passed by the command line or with more custom changes by modifying the underlying Python code.

To start using it, first install the dependencies in your desired Python enviroment.

To do it with pip:
```bash
pip install -r requirements.txt
```

To do it with conda:
```bash
conda install -c conda-forge --yes --file requirements.txt
```

## Examples

Generate a fairly robust language detector with a small vocabulary by using words as tokens

```bash
python langdetect.py --input data/dataset.csv --voc_size 10 --by_lang --classifier RF --analyzer word --preprocess WL
```

Generate a very robust language detector with a bigger vocabulary by using unigrams and bigrams without generating any plot

```bash
python langdetect.py -i data/dataset.csv -v 1000 -c SVM -a char -n 1 2 --no_plot
```