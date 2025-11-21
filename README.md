# CS109 Challenge

The goal is no imports (except for numpy). This whole nnet is self-contained.

Data distribution:
<img width="1512" height="982" alt="image" src="https://github.com/user-attachments/assets/285da8e4-304a-4a2a-8729-1818ea295f5b" />

- Update 11/21; model trains okay on the above dataset; problems remain:
  - Overfits to Brazil, Europe, Australia band; most of the data from the dataset appears in those spots so the model loves to just make a large splotch of probability density covering that band and refuse to make specific guesses
  - My model does KL-divergence; when calculating the loss with `target * np.log(inputs)` for loss, in the parts of the target distribution where `target` is 0, the model is not penalized for this. How should I penalize?
  - Also tends to overfit a TON to the training set; seeing good results for the first ~10 epochs lowering both training loss and test loss, but after that the loss graphs diverge a ton. The training loss almost compresses it's knowledge down to optimize for the training set only; as in capitalizing on the fact most images are from Brazil/Europe/Australia and only guessing there. As a result, it just gives up on guessing everywhere else.
  - The splotches are pretty big; my `sigma` value for the normal distribution around the target location when training for 80 epochs was 2, which creates a blob around the size of America.
  - Will update to the new dataset and see how it goes, will be more accurate.