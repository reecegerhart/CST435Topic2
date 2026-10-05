# Income Insight

Income Insight is a cloud based neural network application that predicts whether an individual's annual income is above or below $50K using the UCI Adult Income dataset. The application uses a PyTorch multilayer perceptron with a reproducible scikit learn preprocessing pipeline and is deployed across three cloud services. Streamlit provides the user interface, FastAPI hosts the machine learning model and prediction API, and Supabase stores the Adult Income data, model runs, artifacts, and prediction audit records. The application also provides model performance analysis, fairness auditing, batch prediction, and model documentation.

## Live Deployment

| Tier | Platform                  | URL                                                       |
| ---- | ------------------------- | --------------------------------------------------------- |
| UI   | Streamlit Community Cloud | https://cst435topic2-azzzknyfkzokhdjk2b4fgy.streamlit.app/ |
| API  | Render                    | https://cst435topic2.onrender.com                         |
| Data | Supabase                  | https://cvjmrqlhdnjuxbkczwrn.supabase.co                  |

The Supabase project reference is `cvjmrqlhdnjuxbkczwrn`.

## Product Purpose

Income Insight demonstrates how a neural network can be used to perform binary classification on real world tabular data. The model uses demographic, employment, education, and financial features from the UCI Adult Income dataset to estimate whether an individual earns more than $50K per year.

The project was designed as a complete cloud based machine learning product rather than only a model training exercise. Users can submit individual records or CSV files for prediction, examine model performance, compare trained configurations, review feature importance, and examine fairness metrics across protected groups.

Because income prediction can have important consequences for different groups, the application also provides information about model limitations and fairness. The system is intended primarily as an educational and analytical demonstration and should not be used as the sole basis for consequential decisions about individuals.

## Problem Statement

The goal of Income Insight is to develop a neural network capable of classifying whether an individual's annual income is at or below $50K or above $50K. The project uses the UCI Adult Income dataset, which contains demographic, educational, employment, and financial information.

This problem represents a binary classification task because every record belongs to one of two income classes. The challenge is that the input data contains both numerical and categorical features, missing values, and demographic characteristics that can reflect existing disparities within the underlying data.

The solution combines a scikit learn preprocessing pipeline with a PyTorch multilayer perceptron. The preprocessing pipeline converts the raw dataset into numerical features suitable for the neural network. The trained model then produces both an income classification and a probability for the predicted class.

## Dataset

The application uses the UCI Adult Income dataset rather than the synthetic dataset included in the original template. The dataset contains more than 30,000 records and uses a binary income target consisting of `<=50K` and `>50K`.

The dataset was downloaded and loaded into the project's Supabase `adult_income` table. Each row represents one individual record used for model development and evaluation.

| Feature          | Type        | Description                           |
| ---------------- | ----------- | ------------------------------------- |
| `age`            | Numeric     | Age of the individual                 |
| `workclass`      | Categorical | Type of employment                    |
| `fnlwgt`         | Numeric     | Census sampling weight                |
| `education`      | Categorical | Highest education level               |
| `education_num`  | Numeric     | Numerical representation of education |
| `marital_status` | Categorical | Marital status                        |
| `occupation`     | Categorical | Occupation                            |
| `relationship`   | Categorical | Relationship status                   |
| `race`           | Categorical | Reported race                         |
| `sex`            | Categorical | Reported sex                          |
| `capital_gain`   | Numeric     | Capital gains                         |
| `capital_loss`   | Numeric     | Capital losses                        |
| `hours_per_week` | Numeric     | Hours worked per week                 |
| `native_country` | Categorical | Country of origin                     |
| `income`         | Target      | Income classification                 |

The dataset is divided into training, validation, and test sets. The training data is used to fit the preprocessing pipeline and neural network. The validation data is used to select the best model checkpoint and configuration. The test data remains held out until final evaluation.

## Algorithm of Solution

The solution uses a supervised neural network classification approach. Raw Adult Income records are first processed by a reproducible scikit learn pipeline. Numerical features are imputed and standardized, while categorical features are imputed and one hot encoded.

The transformed features are then passed to a PyTorch multilayer perceptron. The network contains multiple fully connected layers followed by nonlinear activation functions. The final layer produces a probability used to determine whether the predicted income class is `<=50K` or `>50K`.

The model is trained using the Adam optimizer and binary cross entropy loss. Multiple configurations are trained so that activation function, dropout, and network depth can be evaluated. Validation performance is used to select the best checkpoint, while the test data is reserved for final evaluation.

After training, the preprocessing pipeline and neural network are serialized together. The FastAPI service loads these artifacts and applies the same preprocessing operations whenever a new prediction is requested.

## Architecture

```text
                         HTTPS

                           │

                           ▼

┌──────────────────────┐        ┌──────────────────────────┐
│ Streamlit Cloud     │        │ FastAPI on Render       │
│                     │        │                         │
│ ui/app.py           │──────► │ Prediction API          │
│ Thin client         │◄────── │ PyTorch MLP             │
│ Score a Row         │        │ sklearn preprocessing   │
│ Score a CSV         │        │ Model artifacts         │
│ Performance         │        │ Fairness audit          │
│ Bias Audit          │        │                         │
│ Model Card          │        └────────────┬────────────┘
└──────────────────────┘                     │
                                             │ Service role
                                             ▼
                                  ┌────────────────────────┐
                                  │ Supabase               │
                                  │                        │
                                  │ adult_income           │
                                  │ runs                   │
                                  │ run_artifacts          │
                                  │ predictions            │
                                  └────────────────────────┘
```

Income Insight uses a three cloud architecture consisting of Streamlit Community Cloud, Render, and Supabase. Streamlit Community Cloud hosts the user interface and acts as a thin client. It sends prediction requests to the FastAPI service rather than containing the machine learning model itself.

Render hosts the FastAPI backend. FastAPI receives prediction requests, validates the input data, applies the serialized scikit learn preprocessing pipeline, and passes the processed data to the PyTorch multilayer perceptron. The API returns the predicted income class and probability to Streamlit. FastAPI also handles prediction logging and communicates with Supabase using service role credentials.

Supabase provides persistent storage for the Adult Income dataset, model runs, serialized model artifacts, and prediction audit records. This separation allows the user interface, machine learning service, and persistent data layer to operate independently while functioning together as one application.

## Neural Network

The original template used a single hidden layer, so the model was expanded into a multilayer perceptron for Income Insight. The selected baseline model contains two hidden layers with 128 and 64 neurons. ReLU activation is applied after each hidden layer, and dropout with a rate of 0.20 is used as a regularization technique.

```text
Input Features

      │
      ▼
Hidden Layer 1
128 neurons
      │
     ReLU
      │
   Dropout
      │
      ▼
Hidden Layer 2
64 neurons
      │
     ReLU
      │
      ▼
Output Layer
      │
      ▼
Income Prediction
```

The architecture is configurable through YAML files in `api/configs/`. The project supports ReLU, GELU, Tanh, and Leaky ReLU activation functions. It also supports different hidden layer configurations and dropout settings.

The deeper configuration uses four hidden layers with 256, 128, 64, and 32 neurons. These configurations allow the project to evaluate how network depth and regularization affect performance.

The model is implemented using PyTorch and trained with the Adam optimizer and binary cross entropy loss.

## MLP Training Process

The Income Insight multilayer perceptron is trained through repeated forward propagation and backpropagation across multiple epochs. During each training step, the preprocessed training data is passed through the network one batch at a time.

During forward propagation, the input matrix is multiplied by the weights of each layer and combined with the corresponding bias values. The resulting values are passed through the activation function before being provided to the next layer. For a hidden layer, the calculation can be represented as:

$$
Z^{(l)} = W^{(l)}A^{(l-1)} + b^{(l)}
$$

$$
A^{(l)} = ReLU(Z^{(l)})
$$

The final layer produces a probability representing the predicted likelihood that the individual's income is above $50K. Binary cross entropy is used as the cost function to measure the difference between the predicted probabilities and the actual income labels.

After calculating the cost, backpropagation calculates how much each model parameter contributed to the error. The derivatives of the cost with respect to the weights and biases are calculated using the chain rule. These gradients are then used by the Adam optimizer to update the model parameters.

$$
W^{(l)} = W^{(l)} - \eta \frac{\partial J}{\partial W^{(l)}}
$$

The training process repeats these forward propagation, cost calculation, backpropagation, and parameter update steps across multiple batches and epochs. The model therefore learns its weights gradually as it minimizes the training loss and improves its predictions.

After training, the network performs forward propagation on unseen test records to produce output probabilities. A threshold of 0.50 is applied to the probability to obtain the final predicted class. A probability below 0.50 is classified as `<=50K`, while a probability of 0.50 or greater is classified as `>50K`.

For the selected model, this process produced 85.8 percent test accuracy and a ROC AUC of 0.910. The results indicate that the trained MLP learned useful relationships between the input features and income classification, although the confusion matrix shows that the model has more difficulty correctly identifying the >50K class.


## Preprocessing

The preprocessing pipeline is implemented using scikit learn and is fitted using the training data only. This prevents information from the validation and test sets from influencing the preprocessing process and helps prevent data leakage.

Numeric features are processed by replacing missing values with the median value calculated from the training data. The numeric features are then standardized using `StandardScaler` so that features with different numerical ranges can be provided to the neural network on a more consistent scale.

Categorical features are processed by replacing missing values with `Unknown` and then converting categorical values into numerical representations using one hot encoding. This allows the neural network to process features such as work class, education, marital status, occupation, and native country.

The complete preprocessing pipeline is serialized together with the trained neural network. When a prediction is made, the same preprocessing steps used during training are applied to the new input data. Keeping the preprocessing pipeline with the model ensures that training and prediction use the same feature transformations.

## Training

Training is performed through the command line using `api/train.py` rather than through a long running HTTP request. This keeps model training separate from the deployed prediction service and allows different configurations to be trained and compared independently.

The baseline model can be trained with:

```bash
python -m api.train --config api/configs/default.yaml --activate
```

Additional configurations can be trained with:

```bash
python -m api.train --config api/configs/gelu.yaml
python -m api.train --config api/configs/no_dropout.yaml
python -m api.train --config api/configs/deep.yaml
```

The training process uses separate training, validation, and test data. The validation set is used to select the best model checkpoint, while the test set remains separate for final evaluation. Early stopping is used to stop training when validation performance stops improving.

Each completed training run stores its configuration and performance metrics in Supabase. The trained PyTorch model and preprocessing pipeline are serialized as model artifacts so the FastAPI service can load the selected model for prediction.

## Model Comparison

The project uses multiple configurations to perform a controlled comparison of neural network designs. The configurations vary activation function, dropout, and network depth while keeping the dataset and evaluation process consistent.

The baseline ReLU model uses two hidden layers with 128 and 64 neurons and a dropout rate of 0.20. The GELU configuration uses the same architecture while changing the activation function to GELU. The no dropout configuration removes dropout from the baseline architecture. The deep configuration uses four hidden layers with 256, 128, 64, and 32 neurons.

The completed model runs produced the following results:

| Configuration | Accuracy | Precision | Recall |    F1 | ROC AUC |   ECE |
| ------------- | -------: | --------: | -----: | ----: | ------: | ----: |
| Baseline ReLU |    0.858 |     0.749 |  0.612 | 0.673 |   0.910 | 0.007 |
| GELU          |    0.858 |     0.757 |  0.600 | 0.669 |   0.910 | 0.009 |
| No Dropout    |    0.857 |     0.725 |  0.646 | 0.683 |   0.911 | 0.013 |
| Deep ReLU     |    0.858 |     0.747 |  0.614 | 0.674 |   0.910 | 0.011 |

The results show that GELU did not meaningfully outperform ReLU. GELU produced slightly higher precision, but recall and F1 were lower. The no dropout configuration produced the highest F1 score and recall, but its calibration was worse than the selected baseline. The deeper model produced only a small F1 improvement without improving accuracy.

The baseline ReLU configuration was selected because it provided a strong overall balance between classification performance and probability calibration. Its expected calibration error was 0.007, compared with 0.009 for GELU, 0.013 for no dropout, and 0.011 for the deeper configuration.

Each completed configuration is stored as a separate run in Supabase. Model selection was based on held out performance rather than simply choosing the configuration with the highest training accuracy.

## API Endpoints

| Method | Endpoint         | Purpose                                              |
| ------ | ---------------- | ---------------------------------------------------- |
| `POST` | `/predict`       | Predict income for one record                        |
| `POST` | `/predict_batch` | Predict income for a CSV upload                      |
| `GET`  | `/schema`        | Return the model feature contract                    |
| `GET`  | `/runs`          | List completed model runs                            |
| `GET`  | `/runs/{run_id}` | Retrieve a specific model run                        |
| `GET`  | `/audit`         | Return fairness metrics by protected attribute       |
| `GET`  | `/healthz`       | Check API and database health                        |
| `GET`  | `/version`       | Return application and framework version information |

The `/predict` endpoint validates the supplied feature schema before making a prediction. Predictions return both the predicted class and probability.

Each prediction is recorded in Supabase with a hash of the input features, predicted label, predicted probability, model run, and timestamp. This provides an audit trail of prediction activity without storing the complete input record in the prediction table.

The `/predict_batch` endpoint accepts CSV files and returns a prediction for each input row. The application validates the required columns before processing the file.

## Streamlit Application

The Streamlit application provides the user interface for the deployed product. It acts as a thin client and sends prediction requests to the FastAPI service rather than loading the machine learning model directly.

### Concepts

The Concepts tab explains the mathematics behind neural networks using matrix notation. It includes forward propagation, backpropagation, the cost function, weight updates, and a worked XOR example.

### Score a Row

The Score a Row tab creates the input form from the feature schema returned by the FastAPI `/schema` endpoint. A user can enter one individual's information and receive an income classification and probability.

### Score a CSV

The Score a CSV tab allows users to upload a CSV containing multiple records. The records are sent to `/predict_batch`, and the resulting predictions can be downloaded as a CSV file.

### Model Performance

The Model Performance tab displays the selected model's training and validation curves, confusion matrix, per class precision, recall, and F1 scores, calibration information, permutation importance, and completed model comparisons.

### Bias Audit

The Bias Audit tab displays false positive and false negative rates by protected attributes such as sex and race. These results allow model performance to be examined across different groups.

### Model Card

The Model Card documents the intended purpose of the model, dataset, architecture, performance, limitations, fairness concerns, and responsible use considerations.

## Model Evaluation

The active baseline ReLU model achieved 85.8 percent accuracy, 74.9 percent precision, 61.2 percent recall, an F1 score of 67.3 percent, and a ROC AUC of 0.910.

The confusion matrix provides additional information about the errors made by the model.

|              | Predicted <=50K | Predicted >50K |
| ------------ | --------------: | -------------: |
| Actual <=50K |            5215 |            359 |
| Actual >50K  |             681 |           1072 |

The model correctly classified 5,215 individuals in the <=50K class and 1,072 individuals in the >50K class. It produced 359 false positives and 681 false negatives.

The larger number of false negatives shows that the >50K class is more difficult for the model to identify. Its recall for the >50K class is approximately 61.2 percent, meaning that the model incorrectly classifies a substantial portion of people who actually belong to that class. This demonstrates why overall accuracy alone is not sufficient for evaluating the model.

Calibration is evaluated separately so that predicted probabilities can be compared with observed outcomes. The selected model has an expected calibration error of 0.007, indicating relatively good probability calibration.

Permutation importance was used to examine which features the model relies on most heavily. Marital status had the highest importance at approximately 0.0671, followed by capital gain at 0.0398, education number at 0.0375, and age at 0.0258.

Occupation had an importance of approximately 0.0168, while hours per week had an importance of approximately 0.0149. Relationship, capital loss, work class, and native country had lower importance values.

Permutation importance indicates which features the model relies on when making predictions. It does not prove that a feature directly causes differences in income.

## Fairness and Bias

Income prediction involves features that can reflect demographic and social differences in the underlying dataset. The application therefore includes a bias audit that calculates false positive and false negative rates for protected groups.

A false positive occurs when the model predicts an income above $50K when the actual income is at or below $50K. A false negative occurs when the model predicts an income at or below $50K when the actual income is above $50K.

For the active baseline model, the audit by sex produced a false positive rate of approximately 2.4 percent for females and 8.8 percent for males. The false negative rate was approximately 42.9 percent for females and 38.0 percent for males.

This corresponds to a false positive rate gap of approximately 6.4 percentage points and a false negative rate gap of approximately 4.9 percentage points.

These results show that the model does not make the same types of errors at the same rates across the groups. Males experience the higher false positive rate, while females experience the higher false negative rate.

The higher female false negative rate is particularly important because it means the model is more likely to incorrectly classify a woman whose actual income is above $50K as being at or below $50K.

A difference in performance between groups does not automatically prove intentional discrimination, but it provides evidence that should be investigated before using the model in a high impact setting. Because income classification can affect consequential decisions, the model should not be treated as an authoritative measure of an individual's economic potential.

## Database

Supabase provides persistent storage for the application.

The primary tables are:

```text
adult_income
runs
run_artifacts
predictions
```

The `adult_income` table stores the Adult Income dataset.

The `runs` table stores model architecture, hyperparameters, training information, and evaluation metrics.

The `run_artifacts` table stores serialized model and preprocessing artifacts.

The `predictions` table stores prediction audit information including the request hash, predicted label, predicted probability, model run, and timestamp.

Row level security is enabled on the database tables. The FastAPI service uses the Supabase service role for operations requiring write access, while the Streamlit application uses the Supabase anonymous key only for permitted read operations.

## Testing and Reliability

The project includes an automated Pytest suite covering API behavior, regression behavior, and Supabase integration.

The `test_apy.py` tests cover health checks, version information, schema validation, invalid prediction inputs, valid predictions, batch prediction row counts, and audit validation.

The `test_regression.py` tests use a frozen reference row to verify that the model produces a stable probability through both the prediction function and the API. The required tolerance for the frozen probability is ±0.001.

The `test_supabase_roundtrip.py` test verifies that a successful `/predict` request creates a corresponding row in the Supabase `predictions` table.

The complete test suite contains 14 tests, and all 14 tests currently pass.

## Model Strengths and Weaknesses

The primary strength of the Income Insight model is its overall predictive performance. The model achieved 85.8 percent accuracy and a 0.910 ROC AUC while using a reproducible preprocessing pipeline and a multilayer perceptron. The project also evaluates multiple model configurations instead of relying on a single architecture, and automated testing helps ensure that application changes do not unexpectedly affect predictions. The cloud architecture further demonstrates how the model, API, database, and user interface can operate together as a complete application.

The primary weakness is that performance is not equal across the two income classes. The model achieved 61.2 percent recall for the >$50K class and produced 681 false negatives compared with 359 false positives. The fairness audit also identified differences between males and females. Females had a higher false negative rate of 42.9 percent, while males had a higher false positive rate of 8.8 percent.

These results show that overall accuracy does not fully represent the model's limitations. The model should therefore be treated as an analytical demonstration rather than an authoritative system for making consequential decisions about individuals.

## Individual Contributions

This project was completed independently, so all project responsibilities were performed by me.

* Selected and prepared the UCI Adult Income dataset.

* Developed the Supabase database structure and migrations.

* Implemented the scikit learn preprocessing pipeline.

* Developed and trained the PyTorch multilayer perceptron.

* Created configurable model experiments for activation functions, dropout, and network depth.

* Evaluated model accuracy, precision, recall, F1 score, ROC AUC, calibration, confusion matrix, and permutation importance.

* Developed the FastAPI prediction service and API endpoints.

* Implemented prediction logging and request hashing.

* Implemented the fairness audit and group level error analysis.

* Developed the Streamlit user interface and application tabs.

* Created the automated Pytest suite covering API validation, batch prediction, regression behavior, and Supabase integration.

* Deployed the application using Streamlit Community Cloud, Render, and Supabase.

* Completed the Model Card, README documentation, engineering report, model comparison, fairness analysis, and presentation materials.

All code, testing, deployment, documentation, and analysis in this repository represent my individual work.

## Project Structure

```text
CST435Topic2/

│
├── README.md
├── MODEL_CARD.md
│
├── shared/
│   └── schemas.py
│
├── api/
│   ├── main.py
│   ├── training.py
│   ├── train.py
│   ├── db.py
│   ├── configs/
│   │   ├── default.yaml
│   │   ├── gelu.yaml
│   │   ├── no_dropout.yaml
│   │   └── deep.yaml
│   └── requirements.txt
│
├── ui/
│   ├── app.py
│   └── requirements.txt
│
├── db/
│   ├── migrations/
│   ├── load.py
│   └── comparison_table.py
│
├── models/
│
├── tests/
│   ├── conftest.py
│   ├── make_fixture.py
│   ├── test_apy.py
│   ├── test_regression.py
│   └── test_supabase_roundtrip.py
│
├── render.yaml
├── requirements-dev.txt
└── .env.example
```

## Deployment

The application is deployed across three cloud services.

Streamlit Community Cloud hosts the user interface and provides the product's interactive frontend.

Render hosts the FastAPI backend, prediction endpoints, preprocessing pipeline, and PyTorch model.

Supabase hosts the Adult Income dataset, model runs, model artifacts, prediction records, and fairness audit information.

The deployed Streamlit application is available at:

https://cst435topic2 azzzknyfkzokhdjk2b4fgy.streamlit.app

The deployed FastAPI service is available at:

https://cst435topic2.onrender.com

The FastAPI health endpoint is available at:

https://cst435topic2.onrender.com/healthz

The Supabase project reference is:

`cvjmrqlhdnjuxbkczwrn`

## Product Presentation

The final product presentation demonstrates the application from end to end. The presentation covers the Streamlit interface, individual prediction, CSV prediction, neural network concepts, model performance, fairness auditing, implementation architecture, and cloud deployment.

Product presentation video links will be included here before final submission.

## Project Status

Income Insight is substantially complete and deployed across Streamlit Community Cloud, Render, and Supabase. The project uses the real UCI Adult Income dataset and includes a PyTorch multilayer perceptron, configurable model architecture, scikit learn preprocessing, multiple training configurations, model comparison, FastAPI prediction endpoints, batch CSV prediction, prediction logging, fairness auditing, and Streamlit performance visualizations.

The required automated testing has also been completed. The project contains 14 Pytest tests covering API validation, batch prediction, regression testing, and Supabase integration, with all 14 tests passing.

Model evaluation and fairness analysis have been completed. The selected model achieves 85.8 percent accuracy, 74.9 percent precision, 61.2 percent recall, 67.3 percent F1, and a 0.910 ROC AUC. The fairness audit has also been completed, including false positive and false negative rates by sex.

The remaining submission work consists primarily of finalizing the Model Card, verifying all README links, adding the product presentation video links, and completing any final documentation checks required before submission.

## Engineering Report

### Decision Justifications

The Income Insight project uses a PyTorch multilayer perceptron to classify whether an individual earns more than $50K per year using information from the UCI Adult Income dataset. The selected model uses two hidden layers with 128 and 64 neurons, ReLU activation, and a dropout rate of 0.20. On the held out test data, the model achieved 85.8 percent accuracy, 74.9 percent precision, 61.2 percent recall, 67.3 percent F1, and a 0.910 ROC AUC.

Several configurations were tested to compare activation functions, network depth, and dropout. The baseline ReLU configuration achieved an F1 score of 0.673. The deeper ReLU configuration achieved a slightly higher F1 score of 0.674, while GELU achieved an F1 score of 0.669. The no dropout configuration achieved the highest F1 score at 0.683 and the highest recall at 64.6 percent, but its expected calibration error was 0.013 compared with 0.007 for the selected baseline.

GELU therefore did not meaningfully outperform ReLU. Although GELU produced slightly higher precision, it produced lower recall and F1, while its calibration was also slightly worse. The baseline ReLU model was selected because it provided a strong overall balance between classification performance and probability calibration.

The confusion matrix provides a clearer view of where the model struggles. The model correctly classified 5,215 individuals in the <=50K class and 1,072 individuals in the >50K class. It produced 359 false positives and 681 false negatives.

Because there were substantially more false negatives than false positives, the model has more difficulty identifying individuals who actually earn more than $50K. Its recall for the >50K class was approximately 61.2 percent. This means that a significant portion of individuals who actually belong to the higher income class were incorrectly classified as earning <=50K.

Permutation importance was used to determine which features the model relied on most when making predictions. Marital status had the highest importance at 0.06708, followed by capital gain at 0.03980, education number at 0.03752, and age at 0.02580. Occupation had an importance of 0.01677, while hours per week had an importance of 0.01485.

These results provide evidence about which features influence the model's predictive behavior. They should not be interpreted as evidence that these features directly cause differences in income.

### Bias and Fairness Reflection

The bias audit evaluated false positive and false negative rates by sex. The model produced a false positive rate of 2.4 percent for females and 8.8 percent for males, resulting in a false positive rate gap of approximately 6.4 percentage points.

The false negative rate was 42.9 percent for females and 38.0 percent for males, resulting in a false negative rate gap of approximately 4.9 percentage points.

These results demonstrate that the model does not make errors equally across the two groups. Males experience more false positives, while females experience more false negatives. The higher female false negative rate means the model is more likely to classify a woman who actually earns above $50K as earning at or below $50K.

Because the Adult dataset contains demographic disparities, these differences should be investigated before using the model in a consequential setting. A difference in error rates does not by itself establish intentional discrimination, but it provides evidence that should not be ignored.

The model should not be used as the sole basis for decisions involving employment, lending, compensation, or other situations where an incorrect prediction could significantly affect a person. Additional validation, fairness analysis, monitoring, and human oversight would be necessary for any consequential use.

### Worldview Reflection

Christian ethics emphasizes impartial judgment as a responsibility rather than simply a preference. Deuteronomy 1:17 states, "You shall not be partial in judgment" (English Standard Version Bible, 2001).

The bias audit shows that women are treated worse with respect to false negatives because 42.9 percent of women who actually earn more than $50K were incorrectly classified, compared with 38.0 percent of men. At the same time, men experience the higher false positive rate. This demonstrates that fairness cannot be represented by a single overall metric because different groups can experience different types of errors.

Before deployment, the development team owes the group experiencing the greater disadvantage careful investigation and responsible action. This includes examining the source of the disparity, testing alternative preprocessing and modeling approaches, monitoring performance across groups, and determining whether the model is appropriate for its intended use.

Human oversight should remain part of any consequential decision, and model predictions should be treated as estimates rather than unquestionable judgments.

### Testing

The project includes automated Pytest tests covering the required API behavior.

The tests validate the schema used by `/predict`, verify that invalid and missing fields are rejected, and confirm that `/predict_batch` returns the same number of predictions as the input rows.

A frozen reference row is used as a regression test to verify that the model probability remains stable within the required tolerance of ±0.001. This protects against unintended changes to the model or preprocessing pipeline.

A Supabase integration test confirms that a successful `/predict` request creates a corresponding row in the `predictions` table.

The complete test suite contains 14 tests, and all 14 tests pass.

### Deployment

Income Insight uses a three cloud architecture. Streamlit Community Cloud provides the user interface, Render hosts the FastAPI backend and machine learning model, and Supabase provides persistent database storage.

Streamlit communicates with FastAPI for prediction requests. FastAPI handles model inference, input validation, prediction logging, and communication with Supabase. Supabase stores the Adult Income data, training runs, model artifacts, prediction records, and audit information.

The deployed application is available through Streamlit Community Cloud and the API is available through Render. The Supabase project reference is `cvjmrqlhdnjuxbkczwrn`.

### Conclusion

Overall, Income Insight demonstrates a complete machine learning application that connects data preparation, neural network training, model evaluation, automated testing, cloud deployment, and fairness analysis.

The selected model achieved strong overall classification performance with 85.8 percent accuracy and a 0.910 ROC AUC. However, the confusion matrix shows that the model has greater difficulty identifying the >$50K class, while the fairness audit shows that women experience a higher false negative rate than men.

These findings demonstrate why responsible machine learning requires more than maximizing accuracy. Before the model could be used for consequential real world decisions, its group level performance should continue to be evaluated, disparities should be investigated, and appropriate human oversight should be maintained.

## References

Dua, D., & Graff, C. (2019). UCI machine learning repository. University of California, Irvine, School of Information and Computer Sciences.

English Standard Version Bible. (2001). Crossway.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high performance deep learning library. *Advances in Neural Information Processing Systems, 32*.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825 to 2830.
