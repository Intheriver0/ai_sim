# Candidate screening

Exploratory internal screening, not independent selected-model performance. No outer scoring.

600 trees per model; K=1; reference G1/G2 fixed.

| Candidate | Months | Mean NMAE | Worst NMAE | Complete hours | W (won) |
|---|---:|---:|---:|---:|---:|
| baseline | 12 | 3.7277% | 5.6906% | 960 | 20,858,380 |
| l1_loss | 12 | 3.5366% | 5.5098% | 960 | 19,473,062 |
| simple_tree | 12 | 3.7815% | 5.5961% | 960 | 20,746,901 |
| recent_2years | 12 | 3.7130% | 5.6814% | 960 | 21,565,480 |
| round17_only | 12 | 3.7340% | 5.6118% | 960 | 20,531,399 |
| extra_trees | 12 | 3.6581% | 5.4907% | 960 | 19,277,926 |
| no_direct_coordinates | 12 | 3.8392% | 5.7884% | 960 | 21,686,299 |
| tiny_tree | 12 | 3.8463% | 5.5289% | 960 | 20,638,142 |
