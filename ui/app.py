"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

A THIN client:
  * scoring (one row or a CSV) is an HTTPS call to the FastAPI service,
  * the fairness numbers on the Bias Audit tab come from FastAPI's /audit,
  * everything read-only (run metrics, curves, the cross-run audit view) is a
    Supabase query with the ANON key. There are no database writes here and no
    torch/sklearn import.

Configuration comes from st.secrets (see ui/.streamlit/secrets.toml.example):
    API_URL              -> your Render.com base URL
    SUPABASE_URL         -> https://<ref>.supabase.co
    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from supabase import create_client

API_URL = st.secrets["API_URL"].rstrip("/")

st.set_page_config(page_title="Income-Insight", page_icon="💼", layout="wide")
st.title("💼 Income-Insight")
st.caption("Streamlit (this UI) → FastAPI (MLP + sklearn) → Supabase (data and audit log).")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def api_request(method: str, path: str, **kwargs):
    """Call the API; show a readable error and return None on failure."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=180, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        st.error(f"The API rejected the request: {detail}")
    except requests.RequestException as exc:
        st.error(
            "Could not reach the API. The free Render tier sleeps when idle and can "
            f"take a minute to wake up; try again shortly. ({exc})"
        )
    return None


@st.cache_data(ttl=60)
def load_runs() -> list:
    """All runs, read straight from Supabase with the anon key."""
    return supabase_anon().table("runs").select("*").order("id", desc=True).execute().data


@st.cache_data(ttl=600)
def feature_schema(run_id: int) -> dict:
    """The /schema contract for a run (raises on failure, so errors are not cached)."""
    resp = requests.get(f"{API_URL}/schema", params={"run_id": run_id}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def pick_run(key: str):
    """A run selector; defaults to the active run. Returns the run dict or None."""
    try:
        runs = load_runs()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read runs from Supabase: {exc}")
        return None
    if not runs:
        st.info("No trained runs yet. Run `python -m api.train` to create one.")
        return None
    labels = {
        r["id"]: f"Run {r['id']} · {r['name']}" + (" (active)" if r["is_active"] else "")
        for r in runs
    }
    default = next((i for i, r in enumerate(runs) if r["is_active"]), 0)
    chosen = st.selectbox(
        "Model run", list(labels), index=default, format_func=labels.get, key=key
    )
    return next(r for r in runs if r["id"] == chosen)


concepts, row_tab, csv_tab, perf_tab, audit_tab, card_tab = st.tabs(
    ["Concepts", "Score a Row", "Score a CSV", "Model Performance", "Bias Audit", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("How the network learns")
    st.markdown(
        "The model is a feed-forward neural network (a multi-layer perceptron) that "
        "estimates the probability that a person's income exceeds \\$50K. Each layer "
        "multiplies its input by a weight matrix, adds a bias vector, and applies a "
        "non-linear activation. Training repeats four steps over many epochs: push the "
        "data forward through the network, measure the error with a cost function, push "
        "that error backward to get derivatives, and nudge the weights downhill."
    )

    st.subheader("Forward propagation")
    st.markdown(
        "Write $a^{(0)} = x$ for the input vector and $L$ for the number of layers. "
        "For each layer $\\ell = 1, \\dots, L-1$ the hidden step is:"
    )
    st.latex(r"z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}, \qquad a^{(\ell)} = \phi\big(z^{(\ell)}\big)")
    st.markdown(
        "where $\\phi$ is the activation (ReLU or GELU here). The last layer produces one "
        "logit, and a sigmoid turns it into a probability. With a batch of $n$ rows stacked "
        "as the columns of $A^{(0)}$, every product becomes one matrix multiplication:"
    )
    st.latex(
        r"Z^{(\ell)} = W^{(\ell)} A^{(\ell-1)} + b^{(\ell)}\mathbf{1}^{\top}, \qquad "
        r"A^{(\ell)} = \phi\big(Z^{(\ell)}\big), \qquad "
        r"\hat{y} = \sigma\big(Z^{(L)}\big) = \frac{1}{1+e^{-Z^{(L)}}}"
    )

    st.subheader("Cost function")
    st.markdown("We measure the error with binary cross-entropy, averaged over the $n$ rows:")
    st.latex(
        r"\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\Big[\,y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big]"
    )

    st.subheader("Backward propagation")
    st.markdown(
        "Backpropagation applies the chain rule from the output layer back to the input. "
        "For a sigmoid output with cross-entropy loss, the error at the output simplifies "
        "to the prediction minus the label. It is then carried back through each weight "
        "matrix and multiplied elementwise ($\\odot$) by the derivative of the activation:"
    )
    st.latex(
        r"\delta^{(L)} = \hat{y} - y, \qquad "
        r"\delta^{(\ell)} = \Big(W^{(\ell+1)\top}\delta^{(\ell+1)}\Big)\odot \phi'\big(z^{(\ell)}\big)"
    )
    st.latex(
        r"\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \delta^{(\ell)}\,a^{(\ell-1)\top}, \qquad "
        r"\frac{\partial \mathcal{L}}{\partial b^{(\ell)}} = \delta^{(\ell)}"
    )
    st.markdown(
        "The update moves each weight against its gradient with learning rate $\\eta$. "
        "Plain gradient descent is shown here; we use Adam, which scales each step by "
        "running averages of the gradient, with weight decay as a mild penalty on large weights:"
    )
    st.latex(r"W^{(\ell)} \leftarrow W^{(\ell)} - \eta\,\frac{\partial \mathcal{L}}{\partial W^{(\ell)}}")

    st.subheader("Worked example: XOR")
    st.markdown(
        "XOR cannot be solved by a single linear layer, which is why hidden layers matter. "
        "Take a network with two inputs, two ReLU hidden units, and one sigmoid output. "
        "Suppose the weights are:"
    )
    st.latex(
        r"W^{(1)} = \begin{bmatrix}1&1\\1&1\end{bmatrix},\; b^{(1)} = \begin{bmatrix}0\\-1\end{bmatrix},\; "
        r"W^{(2)} = \begin{bmatrix}1&-2\end{bmatrix},\; b^{(2)} = -0.5"
    )
    st.markdown("**Forward pass** over all four inputs. The hidden values are $h = \\mathrm{ReLU}(W^{(1)}x + b^{(1)})$:")
    st.latex(
        r"\begin{array}{c|c|c|c|c}"
        r"x & h & z = W^{(2)}h + b^{(2)} & \hat{y} = \sigma(z) & \text{class} \\ \hline"
        r"(0,0) & (0,0) & -0.5 & 0.378 & 0 \\"
        r"(0,1) & (1,0) & 0.5 & 0.622 & 1 \\"
        r"(1,0) & (1,0) & 0.5 & 0.622 & 1 \\"
        r"(1,1) & (2,1) & -0.5 & 0.378 & 0"
        r"\end{array}"
    )
    st.markdown(
        "Thresholding at 0.5 gives the XOR truth table, so these weights solve it. Now one "
        "**backward pass** and update, using the input $x=(1,1)$ with label $y=0$ and "
        "$\\eta = 0.1$. The forward pass gave $h=(2,1)$ and $\\hat{y}=0.3775$, so the loss is "
        "$-\\ln(1-0.3775) = 0.474$."
    )
    st.latex(
        r"\delta^{(2)} = \hat{y}-y = 0.3775,\qquad "
        r"\frac{\partial\mathcal{L}}{\partial W^{(2)}} = 0.3775\,[2,\;1] = [0.755,\;0.378]"
    )
    st.latex(
        r"\delta^{(1)} = \big(W^{(2)\top}\delta^{(2)}\big)\odot\mathrm{ReLU}'(z^{(1)}) = "
        r"\begin{bmatrix}0.3775\\-0.755\end{bmatrix}\odot\begin{bmatrix}1\\1\end{bmatrix}, \qquad "
        r"\frac{\partial\mathcal{L}}{\partial W^{(1)}} = \delta^{(1)}x^{\top} = "
        r"\begin{bmatrix}0.378&0.378\\-0.755&-0.755\end{bmatrix}"
    )
    st.markdown("After the update, the weights become:")
    st.latex(
        r"W^{(1)} \approx \begin{bmatrix}0.962&0.962\\1.076&1.076\end{bmatrix},\; "
        r"b^{(1)} \approx \begin{bmatrix}-0.038\\-0.924\end{bmatrix},\; "
        r"W^{(2)} \approx \begin{bmatrix}0.924&-2.038\end{bmatrix},\; b^{(2)} \approx -0.538"
    )
    st.markdown(
        "Running the forward pass on $(1,1)$ again gives a loss of about 0.243, down from 0.474, "
        "so one step moved the network toward the right answer. Our real model does exactly this "
        "in `api/training.py`, with `loss.backward()` computing these gradients automatically."
    )

    st.subheader("From probability to a decision")
    st.markdown(
        "The network outputs a probability. A threshold function turns it into a class: "
        "predict \\>50K when $\\hat{y} \\ge 0.5$, otherwise \\<=50K. A probability of 0.81 "
        "means the model puts an 81% chance on this person earning over \\$50K. Because raw "
        "network outputs are often overconfident, we also fit **temperature scaling** on a "
        "held-out validation set, dividing the logit by a learned $T$ before the sigmoid, so "
        "that 80% predictions come true about 80% of the time. The Model Performance tab "
        "shows this calibration."
    )

# ---------------------------------------------------------------------------
# Score a Row  (form generated from the API's /schema)
# ---------------------------------------------------------------------------
with row_tab:
    st.header("Score a row")
    run = pick_run("row_run")
    if run:
        try:
            schema = feature_schema(run["id"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load the feature schema from the API: {exc}")
            schema = None

        if schema:
            with st.form("score_row"):
                features: dict = {}
                st.markdown("**Numeric features**")
                cols = st.columns(3)
                for i, feat in enumerate(schema["numeric_features"]):
                    stats = schema["numeric_stats"][feat]
                    features[feat] = cols[i % 3].number_input(
                        feat, min_value=int(stats["min"]), max_value=int(stats["max"]),
                        value=int(stats["median"]), step=1,
                    )
                st.markdown("**Categorical features** (\"Unknown\" means not reported)")
                cols = st.columns(3)
                for i, feat in enumerate(schema["categorical_features"]):
                    features[feat] = cols[i % 3].selectbox(feat, schema["categories"][feat])
                submitted = st.form_submit_button("Score", type="primary")

            if submitted:
                resp = api_request(
                    "POST", "/predict", json={"run_id": run["id"], "features": features}
                )
                if resp:
                    c1, c2 = st.columns(2)
                    c1.metric("Predicted income", resp["income"])
                    c2.metric("P(income > $50K)", f"{resp['proba']:.1%}")
                    st.caption(
                        "The probability is calibrated. This prediction was logged to Supabase "
                        "with the inputs stored only as a hash."
                    )

# ---------------------------------------------------------------------------
# Score a CSV
# ---------------------------------------------------------------------------
with csv_tab:
    st.header("Score a CSV")
    run = pick_run("csv_run")
    if run:
        try:
            schema = feature_schema(run["id"])
            template = pd.DataFrame(
                columns=schema["numeric_features"] + schema["categorical_features"]
            )
            st.download_button(
                "Download a blank template (column names only)",
                template.to_csv(index=False), "template.csv", "text/csv",
            )
            st.caption(
                "Required numeric columns: " + ", ".join(schema["numeric_features"])
                + ". Categorical columns may be left blank."
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load the schema for the template: {exc}")

        upload = st.file_uploader("Upload a CSV", type="csv")
        if upload is not None:
            df = pd.read_csv(upload)
            st.write(f"{len(df)} rows uploaded.")
            st.dataframe(df.head(), use_container_width=True)
            if st.button("Score this CSV", type="primary"):
                with st.spinner("Scoring on the FastAPI service..."):
                    resp = api_request(
                        "POST", "/predict_batch", params={"run_id": run["id"]},
                        files={"file": (upload.name, upload.getvalue(), "text/csv")},
                    )
                if resp:
                    preds = resp["predictions"]
                    if len(preds) != len(df):
                        st.warning("Row count mismatch between upload and results.")
                    out = df.copy()
                    out["predicted_income"] = [p["income"] for p in preds]
                    out["predicted_proba"] = [p["proba"] for p in preds]
                    st.success(f"Scored {resp['n_rows']} rows with run {resp['run_id']}.")
                    st.dataframe(out, use_container_width=True)
                    st.download_button(
                        "Download predictions CSV", out.to_csv(index=False),
                        "predictions.csv", "text/csv",
                    )

# ---------------------------------------------------------------------------
# Model Performance  (anon-key read of the runs table)
# ---------------------------------------------------------------------------
with perf_tab:
    st.header("Model performance")
    st.caption("Held-out test split. Read from Supabase with the anon key.")
    run = pick_run("perf_run")
    if run:
        m = st.columns(5)
        for col, (label, key) in zip(
            m, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                ("F1", "f1"), ("ROC-AUC", "roc_auc")],
        ):
            col.metric(label, f"{run[key]:.3f}")
        st.caption(
            f"{run['name']}: hidden layers {run['hidden_sizes']}, {run['activation']}, "
            f"dropout {run['dropout']}. Best epoch {run['best_epoch']}."
        )

        hist = run.get("history")
        if hist:
            curves = pd.DataFrame(hist, index=range(1, len(hist["train_loss"]) + 1))
            curves.index.name = "epoch"
            c1, c2 = st.columns(2)
            c1.subheader("Loss")
            c1.line_chart(curves[["train_loss", "val_loss"]])
            c2.subheader("Accuracy")
            c2.line_chart(curves[["train_acc", "val_acc"]])

        cm = run.get("confusion_matrix")
        if cm:
            st.subheader("Confusion matrix")
            cm_df = pd.DataFrame(
                cm, index=["Actual <=50K", "Actual >50K"],
                columns=["Predicted <=50K", "Predicted >50K"],
            )
            c1, c2 = st.columns(2)
            c1.dataframe(cm_df, use_container_width=True)
            per_class = run.get("per_class")
            if per_class:
                c2.dataframe(pd.DataFrame(per_class).T, use_container_width=True)
                harder = min(per_class, key=lambda k: per_class[k]["recall"])
                st.caption(
                    f"The model finds the {harder} class harder: its recall is "
                    f"{per_class[harder]['recall']:.1%}, versus "
                    f"{max(v['recall'] for v in per_class.values()):.1%} for the other class."
                )

        cal = run.get("calibration")
        if cal:
            st.subheader("Calibration (reliability) plot")
            before = {b["bin_lo"]: b["frac_pos"] for b in cal["before"]}
            after = {b["bin_lo"]: b["frac_pos"] for b in cal["after"]}
            bins = sorted(set(before) | set(after))
            rel = pd.DataFrame(
                {
                    "Perfect": [lo + 0.05 for lo in bins],
                    "Before calibration": [before.get(lo) for lo in bins],
                    "After calibration": [after.get(lo) for lo in bins],
                },
                index=[round(lo + 0.05, 2) for lo in bins],
            )
            rel.index.name = "predicted probability (bin midpoint)"
            st.line_chart(rel)
            st.caption(
                f"Expected calibration error: {run['ece_uncalibrated']:.4f} before, "
                f"{run['ece']:.4f} after (temperature T = {run['temperature']:.2f}). "
                "The closer a curve sits to the Perfect line, the more the probabilities can be trusted."
            )

        imp = run.get("permutation_importance")
        if imp:
            st.subheader("Permutation importance")
            series = pd.Series({k: v["mean"] for k, v in imp.items()}).sort_values(ascending=False)
            st.bar_chart(series)
            st.caption("Drop in test ROC-AUC when one feature's values are shuffled. Bigger means the model relies on it more.")

        with st.expander("Compare all runs"):
            runs_df = pd.DataFrame(load_runs())[
                ["id", "name", "hidden_sizes", "activation", "dropout", "best_epoch",
                 "accuracy", "precision", "recall", "f1", "roc_auc", "ece"]
            ]
            st.dataframe(runs_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Bias Audit  (/audit from the API + anon-key query on the audit_rates view)
# ---------------------------------------------------------------------------
with audit_tab:
    st.header("Bias audit")
    st.markdown(
        "Error rates on the held-out test rows, split by a protected attribute. The "
        "**false-positive rate** is the share of people earning \\<=50K whom the model "
        "labels \\>50K. The **false-negative rate** is the share of people earning "
        "\\>50K whom the model labels \\<=50K. Large gaps between groups are the signal "
        "to investigate. Small groups give noisy rates, so check the `n` column."
    )
    attribute = st.radio("Protected attribute", ["sex", "race"], horizontal=True)

    st.subheader("From the API (/audit)")
    run = pick_run("audit_run")
    if run:
        resp = api_request("GET", "/audit", params={"run_id": run["id"], "by": attribute})
        if resp and resp["rows"]:
            audit_df = pd.DataFrame(resp["rows"]).set_index("group")
            st.dataframe(
                audit_df[["n", "positives", "fp", "fn", "fpr", "fnr", "selection_rate"]],
                use_container_width=True,
            )
            st.bar_chart(audit_df[["fpr", "fnr"]])
            gaps = st.columns(2)
            gaps[0].metric("FPR gap (max - min)", f"{audit_df['fpr'].max() - audit_df['fpr'].min():.3f}")
            gaps[1].metric("FNR gap (max - min)", f"{audit_df['fnr'].max() - audit_df['fnr'].min():.3f}")
        elif resp is not None:
            st.info("No audit data for this run yet. `python -m api.train` logs it automatically.")

    st.subheader("Across all runs (Supabase anon key)")
    try:
        rows = (
            supabase_anon().table("audit_rates").select("*").eq("attribute", attribute)
            .execute().data
        )
        if rows:
            all_df = pd.DataFrame(rows)
            metric = st.selectbox(
                "Metric", ["fpr", "fnr", "selection_rate"], key="audit_metric"
            )
            pivot = all_df.pivot_table(index="run_id", columns="grp", values=metric)
            st.dataframe(pivot, use_container_width=True)
            st.bar_chart(pivot)
        else:
            st.info("No audit rows yet.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read audit_rates from Supabase: {exc}")

# ---------------------------------------------------------------------------
# Model Card  (rendered from a markdown file in the repo)
# ---------------------------------------------------------------------------
with card_tab:
    st.header("Model Card")
    try:
        with open("MODEL_CARD.md", "r", encoding="utf-8") as fh:
            st.markdown(fh.read())
    except FileNotFoundError:
        st.warning("MODEL_CARD.md not found.")