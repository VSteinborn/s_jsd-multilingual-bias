# S<sub>JSD</sub> Multilingual Gender Bias

This Repository presents the code, data and supplementary material used for the paper "An Information-Theoretic Approach and Dataset for Probing Gender Stereotypes in Multilingual Masked Language Models" (Findings of NAACL 2022)

## Dataset

The dataset consists of edited and translated [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) sentence pairs.
The sentences have been modified according to the suggestions of [Blodgett et al. (2021)](https://www.microsoft.com/en-us/research/uploads/prod/2021/06/The_Salmon_paper.pdf) prior to translation.
Translators were supplied translation instructions in the corresponding instruction sheet.

The dataset consists of five csv files, one for each language. 
The language of the the csv file is indicated by the language code in its file name:

English (en), German (de), Thai (th), Indonesian (id) and Finnish (fi)

The columns of the csv files have the following meanings:

- `ID`: The row in the [CrowS-Pairs dataset](https://github.com/nyu-mll/crows-pairs/blob/master/data/crows_pairs_anonymized.csv) where the original version of the sentence pair may be found.
- `A_en`: The edited english version of the more stereotypical CrowS-Pairs sentence. 
- `B_en`: The edited english version of the less stereotypical CrowS-Pairs sentence. (A swapped variant of `A_en`) 
- `A_x`: The translation of `A_en` into the target language. 
- `B_x`: The translation of `B_en` into the target language. 
- `stereo_antistereo`: The bias direction from the CrowS-Pairs study 

## Scripting

In this work we used Python 3.8.11 with the packages listed in `requirements.txt`.
The required packages may be installed via:

```
pip install -r requirements.txt
```

Subsequently, the script may be run via the following command.

```
python main.py 	
	--input INPUT			path to sentence pairs
	--out_dir OUT_DIR		path to output directory for sentence-level data
	--model {				Model to use in analysis
		bert-multi,				mBERT (cased)
		xlm-roberta,			xlmR (base)
		xlm-roberta-L,			xlmR (large)
		bert,					BERT (base-uncased)
		roberta,				RoBERTa (large)
		albert}					ALBERT (xxlarge-v2)
	[--perturb]				Removes the final character of each sentence
```

Results of the measures will be printed to the terminal, which may be piped using `>>`, for example, to a text file.  

## License

The dataset associated with this paper is based on the [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) dataset, 
which has been licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
Thus, this dataset falls under the same license.
For more information on the construction of the original CrowS-Pairs dataset, please refer to their [paper](https://aclanthology.org/2020.emnlp-main.154/).
