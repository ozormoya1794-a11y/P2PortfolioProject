## Report 

1. Which insights did you gain from your EDA? 
- First thing I noticed after I run an analysis on financial and amount columns is that they are all right skewed, so I need to do a log transform in order to gain a good look at the data.

2. How did you determine which columns to drop or keep? If your EDA informed this process, explain which insights you used to 
determine which columns were not needed. 
- When I load and check the first rowns and columns the dataset, I noticed the columns `nameOrig` and `nameDest` contains name ID for the transactions, which I think would just add noise when I run my model if I retain them in the dataset.
- Also for the `isFlaggedFraud` after doing some analysis, I figured out that out of 8,213 actual fraud cases — the system only flagged 16 of them. That means it missed 8,197 real fraud cases — a 99.8% miss rate. And because a feature that correctly identifies only 16 out of 8,213 fraud cases contributes essentially nothing to model performance — keeping it would only add confusion. 

3. Which hyperparameter tuning strategy did you use? Grid-search or random-search? Why? 
- Randomized Search CV was chosen over Grid Search CV for hyperparameter tuning. Grid Search exhaustively tests every possible parameter combination — on a dataset of this scale that would require days of computation. Randomized Search instead samples a fixed number of random combinations (n_iter=20) across the parameter space, making it significantly faster while still finding near-optimal parameters.
To further reduce computational time, tuning was performed on a stratified 20% sample (443,265 rows) of the training data rather than the full 2.2 million rows. Stratified sampling was used to ensure the fraud rate of 0.296% was preserved in the sample — maintaining a representative distribution of both classes. The best parameters identified on the sample were then used to retrain the final model on the full training set.

4. How did your model's performance change after discovering optimal hyperparameters? 
- Hyperparameter tuning produced notably different outcomes for each model. For Random Forest model the improvement was modest but meaningful. The tuned model achieved an F1 score of 0.8922 compared to the baseline of 0.8834. More importantly, recall improved from 0.81 to 0.87, meaning the tuned model catches approximately 95 additional fraud cases in the test set. This came at a slight precision cost (0.97 → 0.91) — a worthwhile tradeoff since missing fraud is significantly more costly than generating false alarms.
- However, the tuned Gradient Boosting severely overcorrects — recall improves to 0.99 but precision collapses to 0.20, generating 4 false alarms per real fraud case. Despite poor F1 of 0.3257, the excellent AUC-ROC of 0.9986 suggests the model has strong discriminative ability but requires threshold adjustment to balance precision and recall.

5. What was your final F1 Score?
- My final F1 score for Random Forest is 0.8922, while in Gradient Boosting is 0.3257. Overall the tuned Random Forest was selected as the final model based on its superior F1 score of 0.8922, better precision-recall balance and more stable behavior across baseline and tuned configurations.