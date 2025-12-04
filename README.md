# CS109 Challenge - Geonets

A (sort of) working model for geolocation, entirely built with numpy.

Data distribution:
![](images/earth_dist.png)

- Update 11/21; model trains okay on the above dataset; problems remain:
  - Overfits to Brazil, Europe, Australia band; most of the data from the dataset appears in those spots so the model loves to just make a large splotch of probability density covering that band and refuse to make specific guesses
  - My model does KL-divergence; when calculating the loss with `target * np.log(inputs)` for loss, in the parts of the target distribution where `target` is 0, the model is not penalized for this. How should I penalize?
  - Also tends to overfit a TON to the training set; seeing good results for the first ~10 epochs lowering both training loss and test loss, but after that the loss graphs diverge a ton. The training loss almost compresses it's knowledge down to optimize for the training set only; as in capitalizing on the fact most images are from Brazil/Europe/Australia and only guessing there. As a result, it just gives up on guessing everywhere else.
  - The splotches are pretty big; my `sigma` value for the normal distribution around the target location when training for 80 epochs was 2, which creates a blob around the size of America.
  - Will update to the new dataset and see how it goes, will be more accurate.
  - Adaptive learning rate?

- Update 12/4; added batch norm, dropout, and L2 regularization to combat overfitting; results are mixed:
  - The loss graphs look much better now; train/test loss no longer diverge as badly as before
  - However, the model now just guesses the entire distribution; instead of overfitting to the training set, it just ends up matching the shape of the target distribution
  - Varying the `sigma` value shows that the tighter the Gaussian target, the more the model adheres to the shape of the Earth, but it still just guesses the dataset distribution
  - The guess distributions end up looking like the target distributions; the model learns to approximate the prior instead of making specific guesses

  ![](images/distribution.jpg)

