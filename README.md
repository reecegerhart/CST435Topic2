# Income Insight

Income Insight is a cloud based neural network application that predicts whether an individual's annual income is above or below $50K using the UCI Adult Income dataset. The application uses a PyTorch multilayer perceptron with a reproducible scikit learn preprocessing pipeline and is deployed across three cloud services. Streamlit provides the user interface, FastAPI hosts the machine learning model and prediction API, and Supabase stores the Adult Income data, model runs, artifacts, and prediction audit records. The application also provides model performance analysis, fairness auditing, batch prediction, and model documentation.

## Live Deployment

| Tier | Platform                  | URL                                        |
| ---- | ------------------------- | ------------------------------------------ |
| UI   | Streamlit Community Cloud | `https://cst435topic2-azzzknyfkzokhdjk2b4fgy.streamlit.app/`                       |
| API  | Render                    | `https://cst435topic2.onrender.com`        |
| Data | Supabase                  | `https://cvjmrqlhdnjuxbkczwrn.supabase.co` |

## Product Purpose

Income Insight demonstrates how a neural network can be used to perform binary classification on real world tabular data. The model uses demographic, employment, education, and financial features from the UCI Adult Income dataset to estimate whether an individual earns more than $50K per year. Because income prediction can have important consequences for different groups, the application also provides performance metrics, feature importance information, and fairness measurements by protected attributes.

## Dataset

The application uses the real UCI Adult Income dataset rather than the synthetic dataset included in the original template. The dataset contains more than 30,000 records and uses a binary income target consisting of `<=50K` and `>50K`.

The dataset contains the following features:

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

Missing values are retained during data loading and handled by the preprocessing pipeline rather than being removed from the dataset.

## Architecture

```text
                         HTTPS
┌──────────────────────┐          ┌──────────────────────────┐
│ Streamlit Cloud      │          │ FastAPI on Render       │
│                      │          │                          │
│ ui/app.py            │ ───────► │ Prediction API           │
│ Thin client          │          │ PyTorch MLP              │
│ Score a Row          │ ◄─────── │ sklearn preprocessing    │
│ Score a CSV          │          │ Model artifacts          │
│ Performance          │          │ Fairness audit           │
│ Bias Audit           │          │                          │
│ Model Card           │          └────────────┬─────────────┘
└──────────────────────┘                       │
                                               │ service role
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

Streamlit acts as a thin client and does not contain the machine learning model. FastAPI is responsible for model inference and database writes. Supabase provides persistent storage for the dataset, training runs, model artifacts, and prediction audit information.

## Neural Network

The original template used a single hidden layer. Income Insight now uses a multilayer perceptron with at least two hidden layers.

```text
Input Features
      │
      ▼
Hidden Layer 1
      │
   Activation
      │
      ▼
Hidden Layer 2
      │
   Activation
      │
      ▼
Output Layer
      │
      ▼
Income Prediction
```

The model uses PyTorch for neural network training and Adam optimization with binary cross entropy loss.

The architecture is configurable through files in `api/configs/`. Configuration options include hidden layer sizes, activation function, dropout, learning rate, weight decay, batch size, and number of epochs.

Supported activation functions include ReLU, GELU, Tanh, and Leaky ReLU.

## Preprocessing

The preprocessing pipeline is implemented with scikit learn and is fitted using the training data only.

Numeric features are imputed using the median and standardized using `StandardScaler`. Categorical features are imputed using an `Unknown` value and converted to numerical representations using one hot encoding. The preprocessing pipeline is serialized with the neural network model so the same transformations are used during prediction.

This prevents information from the test data from leaking into the training process.

## Training

Training is performed through the command line using `api/train.py` rather than through a long running HTTP request.

Example:

```bash
python -m api.train --config api/configs/default.yaml --activate
```

Additional configurations can be trained with:

```bash
python -m api.train --config api/configs/gelu.yaml
python -m api.train --config api/configs/no_dropout.yaml
python -m api.train --config api/configs/deep.yaml
```

The training process uses separate training, validation, and test data. The validation set is used to select the best checkpoint so that the test set remains available for final evaluation.

Each completed training run stores its configuration and performance information in Supabase. The trained model and preprocessing pipeline are also serialized so the FastAPI service can load the model for predictions.

## Model Comparison

The project uses multiple configurations to perform a controlled comparison of different neural network designs.

The configurations compare factors such as activation functions, dropout, and network depth.

The comparison includes:

```text
Default
Two hidden layers
ReLU
Dropout

GELU
Two hidden layers
GELU
Dropout

No Dropout
Two hidden layers
ReLU
No dropout

Deep
Additional hidden layers
ReLU
Dropout
```

Each configuration is stored as a separate run in Supabase. The final model is selected based on held out performance rather than simply choosing the configuration with the highest training accuracy.

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

Each prediction is recorded in Supabase with a hash of the input features, the predicted label, predicted probability, and the model run used to serve the prediction.

The `/predict_batch` endpoint accepts CSV files and returns a prediction for each input row.

## Streamlit Application

The Streamlit application contains the required product functionality.

### Concepts

The Concepts tab explains the neural network mathematics using matrix notation. It includes forward propagation, backpropagation, the cost function, weight updates, and a worked XOR example.

### Score a Row

The Score a Row tab creates the input form from the feature schema returned by the FastAPI `/schema` endpoint. A user can enter one individual's information and receive an income classification and probability.

### Score a CSV

The Score a CSV tab allows users to upload a CSV containing multiple records. The records are sent to `/predict_batch`, and the resulting predictions can be downloaded as a CSV file.

### Model Performance

The Model Performance tab displays the results of the selected training run, including training and validation curves, the confusion matrix, per class precision, recall, and F1 scores, calibration information, permutation importance, and comparisons between completed model runs.

### Bias Audit

The Bias Audit tab displays false positive and false negative rates by protected attribute such as sex and race. These results allow the performance of the model to be examined across different groups.

### Model Card

The Model Card documents the intended purpose of the model, dataset, architecture, performance, limitations, fairness concerns, and responsible use considerations.

## Model Evaluation

The model is evaluated using multiple classification metrics rather than accuracy alone.

The primary metrics include accuracy, precision, recall, F1 score, and ROC AUC. The application also records the confusion matrix and per class performance.

Calibration is evaluated separately so that the predicted probabilities can be compared with observed outcomes. This is important because a probability such as `0.80` should represent a meaningful estimate of an 80 percent likelihood rather than simply a classification score.

Permutation importance is also used to examine which input features have the greatest effect on model performance.

## Fairness and Bias

Income prediction involves features that may reflect demographic and social differences in the underlying dataset. The application therefore includes a bias audit that calculates false positive and false negative rates for protected groups.

A false positive occurs when the model predicts an income above $50K when the actual income is at or below $50K. A false negative occurs when the model predicts an income at or below $50K when the actual income is above $50K.

Comparing these rates across groups helps identify whether the model makes certain types of errors more frequently for one group than another.

The fairness results should be considered before using the model for real world decisions. A difference in performance between groups does not automatically prove intentional discrimination, but it does provide evidence that should be investigated before deployment in a high impact setting.

## Database

Supabase stores the persistent application data.

The primary tables are:

```text
adult_income
runs
run_artifacts
predictions
```

The `adult_income` table stores the real Adult Income records.

The `runs` table stores model architecture, hyperparameters, training information, and evaluation metrics.

The `run_artifacts` table stores serialized model and preprocessing artifacts.

The `predictions` table stores prediction audit information including the request hash, predicted label, predicted probability, model run, and timestamp.

Row level security is enabled on the database tables. The FastAPI service uses the Supabase service role for operations that require write access, while the Streamlit application uses the Supabase anonymous key only for permitted read operations.

## Testing and Reliability

The project includes an automated Pytest test suite covering the API, regression behavior, and Supabase integration. The `test_apy.py` file contains tests for the FastAPI endpoints, including health checks, version information, schema validation, invalid prediction inputs, valid predictions, batch prediction row counts, and audit validation. The `test_regression.py` file contains frozen reference tests that verify the model produces a stable probability for a known input through both the prediction function and the API. The required tolerance for the frozen probability is ±0.001. The `test_supabase_roundtrip.py` file verifies that a successful prediction is written to the Supabase `predictions` table. The complete test suite currently contains 14 tests, and all 14 tests pass.


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

The complete source code and project history are available in the [Income Insight GitHub repository](https://github.com/reecegerhart/CST435Topic2/tree/main).


## Deployment

The application uses three cloud services.

Streamlit Community Cloud hosts the user interface.

Render hosts the FastAPI service and machine learning model.

Supabase hosts the Adult Income dataset, model runs, model artifacts, and prediction audit data.

The Render service exposes the API at:

`https://cst435topic2.onrender.com`

The API health endpoint can be used to verify the deployment:

`https://cst435topic2.onrender.com/healthz`

## Individual Contributions

This project was completed independently, so all project responsibilities were performed by me. My contributions include selecting and preparing the UCI Adult Income dataset, developing the Supabase database and migrations, implementing the scikit learn preprocessing pipeline, developing and training the PyTorch multilayer perceptron, creating the configurable model experiments, and evaluating the different model configurations.

I also developed the FastAPI backend and its prediction endpoints, implemented prediction logging and fairness auditing, developed the Streamlit user interface, and created the model performance visualizations. I wrote and ran the Pytest test suite, including API validation, batch prediction, regression testing, and Supabase integration testing. I configured the cloud deployment using Streamlit Community Cloud, Render, and Supabase and verified the deployed application.

Finally, I completed the project documentation, Model Card, engineering report, model comparison analysis, fairness analysis, and product presentation materials. All code, testing, deployment, documentation, and analysis in this repository represent my individual work.


## Product Presentation

The final product presentation will demonstrate the application from end to end, including the Streamlit interface, model predictions, performance analysis, fairness audit, neural network concepts, and implementation architecture.

Each team member will provide their own presentation video as required by the assignment.

## Project Status

The Income Insight application is substantially complete and has been deployed across Streamlit Community Cloud, Render, and Supabase. The project now uses the real UCI Adult Income dataset and includes a PyTorch multilayer perceptron with configurable architecture, scikit learn preprocessing, multiple training configurations, model comparison, FastAPI prediction endpoints, batch CSV prediction, prediction logging, fairness auditing, and Streamlit performance visualizations.

The required automated testing has also been completed. The project currently has 14 Pytest tests covering API validation, batch prediction, regression testing, and Supabase integration, with all 14 tests passing. Model evaluation and fairness analysis have been completed using the deployed application. The final model achieves 85.8% accuracy, 74.9% precision, 61.2% recall, 67.3% F1 score, and a 0.910 ROC AUC. The fairness audit has also been completed, including false positive and false negative rates by sex.



## Engineering Report

### Decision Justifications

The Income Insight project uses a PyTorch multilayer perceptron to classify whether an individual earns more than $50K per year using information from the UCI Adult Income dataset. The selected model uses two hidden layers with 128 and 64 neurons, ReLU activation, and a dropout rate of 0.20. On the held out test data, the model achieved 85.8% accuracy, 74.9% precision, 61.2% recall, 67.3% F1 score, and a 0.910 ROC AUC. These results indicate that the model can effectively distinguish between the two income classes, although it has more difficulty identifying individuals who earn more than $50K.

Several configurations were tested to compare activation functions, network depth, and dropout. The baseline ReLU configuration achieved an F1 score of 0.673. The deeper ReLU configuration achieved a slightly higher F1 score of 0.674, while the GELU configuration achieved an F1 score of 0.669. The no dropout configuration achieved the highest F1 score at 0.683 and the highest recall at 64.6%, but its expected calibration error was 0.013 compared with 0.007 for the selected baseline. GELU therefore did not meaningfully outperform ReLU. The baseline ReLU model was selected because it provided a strong balance between classification performance and probability calibration.

The confusion matrix provides a clearer view of where the model struggles. The model correctly classified 5,215 individuals in the <=50K class and 1,072 individuals in the >50K class. It produced 359 false positives and 681 false negatives. Because there were substantially more false negatives than false positives, the model has more difficulty identifying individuals who actually earn more than $50K. Its recall for the >50K class was approximately 61.2%, meaning that a significant portion of individuals in that class were incorrectly classified as earning <=50K. This demonstrates why accuracy alone is not sufficient for evaluating the model.

Permutation importance was used to determine which features the model relied on most when making predictions. Marital status had the highest importance at 0.06708, followed by capital gain at 0.03980, education level at 0.03752, and age at 0.02580. Occupation had an importance of 0.01677, while hours per week had an importance of 0.01485. Relationship, capital loss, work class, and native country had smaller importance values. These results describe which features the model relies on for prediction and should not be interpreted as evidence that the features directly cause differences in income.

### Bias and Fairness Reflection

The bias audit evaluated false positive and false negative rates by sex. The model produced a false positive rate of 2.4% for females and 8.8% for males, resulting in a false positive rate gap of approximately 6.4 percentage points. The false negative rate was 42.9% for females and 38.0% for males, resulting in a false negative rate gap of approximately 4.9 percentage points.

These results demonstrate that the model does not make errors equally across the two groups. Males experience more false positives, meaning the model is more likely to predict an income above $50K for a male whose actual income is <=50K. Females experience more false negatives, meaning the model is more likely to predict <=50K for a female whose actual income is above $50K. The higher false negative rate for females is especially important because it indicates that the model has greater difficulty identifying women who actually belong to the >$50K income class.

Because the Adult dataset contains demographic disparities, these differences should be investigated before using the model in a consequential setting. A difference in error rates does not by itself establish intentional discrimination, but it provides an important warning that should not be ignored. The model should not be used as the sole basis for decisions involving employment, lending, compensation, or other situations where an incorrect prediction could significantly affect a person.

### Worldview Reflection

Christian ethics emphasizes impartial judgment as a responsibility rather than simply a preference. Deuteronomy 1:17 states, "You shall not be partial in judgment" (English Standard Version Bible, 2001). The bias audit shows that women are treated worse with respect to false negatives because 42.9% of women who actually earn more than $50K were incorrectly classified, compared with 38.0% of men. At the same time, men experience the higher false positive rate. Therefore, fairness cannot be represented by a single overall number because different groups can experience different types of errors.

Before deployment, the development team owes the group experiencing the greater disadvantage careful investigation and responsible action. This includes examining the source of the disparity, testing alternative preprocessing and modeling approaches, monitoring performance across groups, and determining whether the model is appropriate for its intended use. Human oversight should remain part of any consequential decision, and model predictions should be treated as estimates rather than unquestionable judgments.

### Testing and Reliability

The project includes automated Pytest tests covering the required API behavior. The tests validate the schema used by `/predict`, verify that invalid and missing fields are rejected, and confirm that `/predict_batch` returns the same number of predictions as the input rows. A frozen reference row is also used as a regression test to verify that the model probability remains stable within the required tolerance of ±0.001. A live Supabase integration test confirms that a successful `/predict` request creates a corresponding row in the predictions table. The complete test suite currently contains 14 tests, and all 14 tests pass.

### Deployment

Income Insight uses a three cloud architecture. Streamlit Community Cloud provides the user interface, Render hosts the FastAPI backend and machine learning model, and Supabase provides persistent database storage. Streamlit communicates with the FastAPI service for predictions while FastAPI handles model inference and database writes. Supabase stores the Adult Income data, training runs, model artifacts, prediction records, and fairness audit information.

The deployed user interface is available at:

https://cst435topic2-azzzknyfkzokhdjk2b4fgy.streamlit.app/

The deployed FastAPI service is available at:

https://cst435topic2.onrender.com

The Supabase project reference is:

`cvjmrqlhdnjuxbkczwrn`

### Conclusion

Overall, Income Insight demonstrates a complete machine learning application that connects data preparation, neural network training, model evaluation, automated testing, cloud deployment, and fairness analysis. The selected model achieved strong overall classification performance with an accuracy of 85.8% and ROC AUC of 0.910. However, the confusion matrix shows that the model has greater difficulty identifying the >$50K class, while the fairness audit shows that women experience a higher false negative rate than men. These findings demonstrate why responsible machine learning requires more than maximizing accuracy. Before the model could be used for consequential real world decisions, its group level performance should continue to be evaluated and appropriate human oversight should be maintained.

### Reference

English Standard Version Bible. (2001). Crossway.


| Run | Configuration | Hidden sizes  | Activation | Dropout | Accuracy | Precision | Recall |    F1 | ROC AUC |   ECE |
| --- | ------------- | ------------- | ---------- | ------- | -------: | --------: | -----: | ----: | ------: | ----: |
| 5   | baseline ReLU | 128/64        | ReLU       | 0.20    |    0.858 |     0.749 |  0.612 | 0.673 |   0.910 | 0.007 |
| 10  | baseline GELU | 128/64        | GELU       | 0.20    |    0.858 |     0.757 |  0.600 | 0.669 |   0.910 | 0.009 |
| 7   | no dropout    | 128/64        | ReLU       | 0       |    0.857 |     0.725 |  0.646 | 0.683 |   0.911 | 0.013 |
| 8   | deep ReLU     | 256/128/64/32 | ReLU       | 0.20    |    0.858 |     0.747 |  0.614 | 0.674 |   0.910 | 0.011 |


