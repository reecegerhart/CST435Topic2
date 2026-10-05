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

## Testing

The project includes a Pytest test suite covering the API contract, prediction behavior, regression behavior, and Supabase integration.

The tests include validation of the `/predict` schema, validation of batch prediction row counts, a frozen reference prediction test, and a live Supabase test that verifies predictions are written to the database.

The frozen reference test helps ensure that changes to the model or preprocessing pipeline do not unexpectedly change a known prediction.

## Project Structure

```text
CST435Topic2/
│
├── README.md
├── MODEL_CARD.md
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
│   ├── test_api.py
│   ├── test_regression.py
│   └── test_supabase.py
│
├── render.yaml
├── requirements-dev.txt
└── .env.example
```

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

Individual contribution evidence should be documented separately for each team member. Contributions should include all significant work performed on the project, including dataset preparation, database development, model development, API development, Streamlit development, testing, deployment, documentation, and presentation work.

## Engineering Report

The final engineering report will discuss the model architecture, preprocessing decisions, activation function comparison, model performance, confusion matrix results, feature importance, calibration, fairness results, and responsible deployment considerations.

The report will also discuss the implications of the model from a Christian worldview, including which group is treated worse by the model and what responsibility the development team has toward that group before deployment.

## Product Presentation

The final product presentation will demonstrate the application from end to end, including the Streamlit interface, model predictions, performance analysis, fairness audit, neural network concepts, and implementation architecture.

Each team member will provide their own presentation video as required by the assignment.

## Project Status

The original three cloud template has been substantially modified into the Income Insight Adult Income classification application.

Completed changes include the real Adult Income dataset, a multilayer neural network with multiple hidden layers, configurable model architecture, scikit learn preprocessing, command line training, model configuration comparisons, FastAPI prediction endpoints, batch CSV prediction, prediction logging, fairness auditing, Streamlit performance visualization, model documentation, and automated testing.

The remaining project work consists primarily of completing the final model comparisons, documenting the final performance and fairness results, completing the Model Card and engineering report, replacing the remaining deployment placeholders, and recording the required presentation videos.
