Lab Exercise
Parse 100 last sentences from dependency treebank using spacy and stanza
are the depedency tags of spacy the same of stanza?
Evaluate against the ground truth the parses using DependencyEvaluator
print LAS and UAS for each parser
BUT! To evaluate the parsers, the sentences parsed by spacy and stanza have to be DependencyGraph objects. To do this , you have to covert the output of the spacy/stanza to ConLL formant, from this format extract the columns following the Malt-Tab format and finally convert the resulting string into a DependecyGraph. Lucky for you there is a library that gets the job done. You have to install the library spacy_conll and use and adapt to your needs the code that you can find below.

OUTPUT:

Spacy Dep Tags:   det compound nsubj aux ROOT det dobj prep det compound pobj advmod nummod npadvmod advmod prep amod prep pcomp amod compound pobj prep det pobj nsubj relcl dobj punct
Stanza Dep Tags:  det compound nsubj aux root det obj case det compound nmod advmod nummod obl:npmod advmod case obl case fixed amod compound obl case det nmod nsubj acl:relcl obj punct
Spacy Dep Tags:   det compound nsubj aux ROOT det dobj prep det compound pobj advmod nummod npadvmod advmod prep amod prep pcomp amod compound pobj prep det pobj nsubj relcl dobj punct
Stanza Dep Tags:  det compound nsubj aux ROOT det obj case det compound nmod advmod nummod obl:npmod advmod case obl case fixed amod compound obl case det nmod nsubj acl:relcl obj punct

Spacy: 
LAS: 0.0
UAS: 0.6926873037236428

Stanza: 
LAS: 0.0
UAS: 0.4647824136384029