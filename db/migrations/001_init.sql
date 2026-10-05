-- 001_init.sql
-- Income-Insight: UCI Adult Income, MLP classifier, fairness audit.
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
--
-- WARNING: this drops the old synthetic-data tables (datasets, runs,
-- run_artifacts, predictions) and recreates everything. That is intended.

drop view  if exists audit_rates;
drop table if exists predictions   cascade;
drop table if exists run_artifacts cascade;
drop table if exists runs          cascade;
drop table if exists datasets      cascade;
drop table if exists adult_income  cascade;

-- ---------------------------------------------------------------------------
-- adult_income: one row per training example (real UCI Adult, ~48.8k rows).
-- Missing values ("?" in the raw data) are stored as NULL; the sklearn
-- pipeline imputes them. `split` is assigned once by db/load.py (stratified
-- 70/15/15, fixed seed) so training, calibration, and the audit all use the
-- same train / val / test partition.
-- sex and race are kept for the bias audit; they are NOT model inputs.
-- ---------------------------------------------------------------------------
create table adult_income (
    id              bigint generated always as identity primary key,
    age             integer not null,
    workclass       text,
    education_num   integer not null,
    marital_status  text,
    occupation      text,
    relationship    text,
    race            text    not null,
    sex             text    not null,
    capital_gain    integer not null,
    capital_loss    integer not null,
    hours_per_week  integer not null,
    native_country  text,
    income          integer not null check (income in (0, 1)),  -- 1 = >50K
    split           text    not null check (split in ('train', 'val', 'test')),
    created_at      timestamptz not null default now()
);

create index idx_adult_income_split on adult_income (split);

-- ---------------------------------------------------------------------------
-- runs: one row per training run. Architecture, hyperparameters, headline
-- metrics, and the diagnostics the Model Performance tab plots.
-- Anon-readable (no model blob here).
-- ---------------------------------------------------------------------------
create table runs (
    id                      bigint generated always as identity primary key,
    name                    text    not null,           -- e.g. "deep-gelu"
    hidden_sizes            integer[] not null,         -- e.g. {128,64}
    activation              text    not null,           -- relu | gelu | ...
    dropout                 double precision not null,
    lr                      double precision not null,
    weight_decay            double precision not null,
    batch_size              integer not null,
    epochs                  integer not null,           -- epochs requested
    best_epoch              integer not null,           -- epoch of best val loss
    -- headline metrics on the held-out TEST split, threshold 0.5
    accuracy                double precision not null,
    precision               double precision not null,
    recall                  double precision not null,
    f1                      double precision not null,
    roc_auc                 double precision not null,
    brier                   double precision,
    ece                     double precision,           -- after calibration
    ece_uncalibrated        double precision,
    temperature             double precision not null default 1.0,
    -- diagnostics (JSONB)
    confusion_matrix        jsonb,                      -- [[tn, fp], [fn, tp]]
    per_class               jsonb,                      -- precision/recall/f1 per class
    history                 jsonb,                      -- per-epoch train/val loss + acc
    calibration             jsonb,                      -- reliability-curve bins
    permutation_importance  jsonb,                      -- feature -> importance
    config                  jsonb,                      -- full config used, for reproducibility
    is_active               boolean not null default false,  -- the run /predict serves
    created_at              timestamptz not null default now()
);

create index idx_runs_created_at on runs (created_at desc);

-- ---------------------------------------------------------------------------
-- run_artifacts: fitted (preprocessor + MLP) blob, base64. Not anon-readable.
-- ---------------------------------------------------------------------------
create table run_artifacts (
    run_id      bigint      primary key references runs (id) on delete cascade,
    model_b64   text        not null,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- predictions: the audit log. Inputs are stored only as a hash.
-- adult_income_id is set when the scored row came from the dataset (e.g. the
-- held-out test rows scored for the audit); it is NULL for ad-hoc rows.
-- ---------------------------------------------------------------------------
create table predictions (
    id               bigint generated always as identity primary key,
    request_hash     text    not null,
    predicted_label  integer not null check (predicted_label in (0, 1)),
    predicted_proba  double precision not null,
    served_by_run_id bigint  not null references runs (id) on delete cascade,
    adult_income_id  bigint  references adult_income (id) on delete set null,
    created_at       timestamptz not null default now()
);

create index idx_predictions_run_id  on predictions (served_by_run_id);
create index idx_predictions_hash    on predictions (request_hash);
create index idx_predictions_adult   on predictions (adult_income_id);

-- ---------------------------------------------------------------------------
-- audit_rates: false-positive and false-negative rates by protected attribute,
-- computed in SQL from predictions JOIN adult_income, on the TEST split only.
-- If a row was scored more than once by the same run, only the latest counts.
--   FPR = FP / (actual <=50K)      FNR = FN / (actual >50K)
-- The view exposes aggregates only, so it is safe to grant to the anon key.
-- ---------------------------------------------------------------------------
create view audit_rates as
with scored as (
    select distinct on (served_by_run_id, adult_income_id)
           served_by_run_id, adult_income_id, predicted_label
    from predictions
    where adult_income_id is not null
    order by served_by_run_id, adult_income_id, created_at desc
)
select
    s.served_by_run_id                                        as run_id,
    g.attribute,
    g.grp,
    count(*)                                                  as n,
    sum((a.income = 1)::int)                                  as positives,
    sum((a.income = 0)::int)                                  as negatives,
    sum((a.income = 0 and s.predicted_label = 1)::int)        as fp,
    sum((a.income = 1 and s.predicted_label = 0)::int)        as fn,
    sum((a.income = 0 and s.predicted_label = 1)::int)::float
        / nullif(sum((a.income = 0)::int), 0)                 as fpr,
    sum((a.income = 1 and s.predicted_label = 0)::int)::float
        / nullif(sum((a.income = 1)::int), 0)                 as fnr,
    avg(s.predicted_label::float)                             as selection_rate
from scored s
join adult_income a on a.id = s.adult_income_id
cross join lateral (values ('sex', a.sex), ('race', a.race)) as g(attribute, grp)
where a.split = 'test'
group by s.served_by_run_id, g.attribute, g.grp;

-- ---------------------------------------------------------------------------
-- Row Level Security.
-- The API uses the service-role key (bypasses RLS). The Streamlit UI uses the
-- anon key and may only read `runs` and the aggregate `audit_rates` view.
-- Everything else has RLS on and no anon policy, so anon sees nothing.
-- ---------------------------------------------------------------------------
alter table adult_income  enable row level security;
alter table runs          enable row level security;
alter table run_artifacts enable row level security;
alter table predictions   enable row level security;

create policy "anon can read runs"
    on runs for select
    to anon
    using (true);

grant select on audit_rates to anon;