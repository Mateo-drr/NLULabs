Lab Exercise
Load another corpus from Gutenberg (e.g. milton-paradise.txt)
Compute descriptive statistics on the reference (.raw, .words, etc.)sentences and tokens.
Compute descriptive statistics in the automatically processed corpus
both with spacy and nltk
Compute lowercased lexicons for all 3 versions (reference, spacy, nltk) of the corpus
compare lexicon sizes
Compute frequency distribution for all 3 versions (reference, spacy, nltk) of the corpus
compare top N frequencies


OUTPUT:

D:\Universidades\Trento\2S\NLU\LAB_01
[nltk_data] Downloading package gutenberg to C:\Users\Mateo-
[nltk_data]     drr\AppData\Roaming\nltk_data...
[nltk_data]   Package gutenberg is already up-to-date!
[nltk_data] Downloading package punkt to C:\Users\Mateo-
[nltk_data]     drr\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
✔ Download and installation successful
You can now load the package via spacy.load('en_core_web_sm')

- Reference Statistics:
Word per sentence 52
Char per word 4
Char per sentence 203
Longest sentence (characters) 2151
Longest sentence (words) 533
Longest word 16

- Spacy Statistics:
Word per sentence 48
Char per word 4
Char per sentence 174
Longest sentence (characters) 1181
Longest sentence (words) 321
Longest word 63

- NLTK Statistics:
Word per sentence 52
Char per word 4
Char per sentence 205
Longest sentence (characters) 2151
Longest sentence (words) 525
Longest word 18

- Reference Lexicon size and 5 top frequencies:
Upper case: 10751 Top 5: {',': 10198, 'and': 2799, 'the': 2505, ';': 2317, 'to': 1758}
Lower case: 9021 Top 5: {',': 10198, 'and': 3395, 'the': 2968, ';': 2317, 'to': 2228}

- Spacy Lexicon size and 5 top frequencies:
Upper case: 10871 Top 5: {'\n': 10425, ',': 10224, 'and': 2796, 'the': 2503, ';': 2303}
Lower case: 9144 Top 5: {'\n': 10425, ',': 10224, 'and': 3392, 'the': 2965, ';': 2303}

- NLTK Lexicon size and 5 top frequencies:
Upper case: 10986 Top 5: {',': 10228, 'and': 2799, 'the': 2505, ';': 2326, 'to': 1758}
Lower case: 9280 Top 5: {',': 10228, 'and': 3391, 'the': 2964, ';': 2326, 'to': 2223}