## General comments (add them to the paper-figure skill)


## General comments for this project only (so do not update the skill)


## Individual figures

figC: remove the embeddings a, b, e. remove c
figD: remove a, b, c, d, f.
figE: remove the nuber of nodes text. For the gallery in b add more examples for each subfactor and add a weighted mean image that activates this factor. feel free to use more space for this then (ie more rows). Clarify which layer factor is being traced back and which is shown, and do not put factor labels on top of images for visibility.
figF: remove c, d, h, i.
figN: a: in L4 with one bar only make that bar thinner. remove the number of nodes text. b: put more emphasis on this by showing more examples, and weighted average images. feel free to spend multiple rows on this. remove d.
figO: remove a, g, h, i.

## Validation figures (I-M)

Restructure this to not have one figure per model, but one figure per validation topic, across models. 

The first figure should be on faithfulness, this includes reconstruction controls, and projection round trip, across models, and the stability across nmf seeds.

The second should be about rand sensitivity, ie the panels e and f of figI across models. 

The third should be about silhouette scores, and also include the weights vs activation advantage on silhouette scores across models. 

remove the comparison to pixel attribution stuff.

feel free to use fig P, Q, R for these as a starting point. 

All of the figures that are being restructured/merged by this should afterwards be cleaned up and deleted such that I only have the new figures. 