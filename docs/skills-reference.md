# Skills Reference

Complete reference for DataPilot's 34 analytical skills across 6 modules: data quality, analysis, query, NLP, visualization, and export.

---

## Data Quality

4 modules for data assessment and preparation in `engine/datapilot/data/`.

### Profiler (`data/profiler.py`)

First skill called on any dataset. Computes per column:
- **Type detection:** numeric, categorical, datetime, boolean, text, id
- **Missing values:** count and percentage
- **Unique values:** count and percentage
- **Statistics (numeric):** mean, std, min, max, quartiles, skewness, kurtosis
- **Distribution (categorical):** top values with counts
- **Duplicate detection:** row-level

**Semantic type detection** (`_semantic_type()`) goes beyond pandas dtypes:

| Condition | Semantic Type |
|-----------|--------------|
| Values in {yes/no, true/false, 0/1} with ≤2 unique | `boolean` |
| >95% unique values, >20 unique | `id` |
| datetime64 dtype | `datetime` |
| Numeric with ≤2 unique | `boolean` |
| Numeric with ≤20 unique or <1% unique ratio | `categorical` |
| Numeric otherwise | `numeric` |
| String with <5% unique ratio | `categorical` |
| String with >50 unique | `text` |
| Default | `categorical` |

**Critical:** Check `is_bool_dtype` BEFORE `is_numeric_dtype` because numpy treats bool as numeric.

### Schema (`data/schema.py`)

Advanced type inference with semantic understanding:

| Type | Detection Method |
|------|-----------------|
| `numeric_continuous` | Numeric with high cardinality |
| `numeric_discrete` | Numeric with low cardinality |
| `categorical_ordinal` | Values match ordinal keywords (low/medium/high) |
| `email` | Regex pattern (≥80% of values match) |
| `phone` | Regex pattern |
| `url` | Regex pattern |

### Validator (`data/validator.py`)

Quality score (0-100) combining: null penalty, duplicate penalty, type consistency, format issues.

- `check_nulls(df)` — Per-column null count/percentage
- `check_duplicates(df, subset=None)` — Duplicate row detection

### Cleaner (`data/cleaner.py`)

- `standardize_column_names(df)` — Lowercase, underscores, strip special chars
- `impute_missing(df, strategy="auto")` — KNN (default), median, mean, or drop
- Duplicate removal, outlier treatment (IQR clipping), type coercion

---

## Query Skills

6 skills for natural language data querying in `engine/datapilot/analysis/query.py` (1059 lines).

### 1. query_data — Filter and Select

Filters rows by conditions. LLM extracts parameters, regex fallback for `where COL = VAL` patterns.

**Safety:** Filter expressions are AST-validated — no function calls allowed.

### 2. pivot_table — Aggregate by Group

Groups data by column with aggregation: mean, sum, count, max, min.

Examples: "Average monthly charges by contract type", "Total revenue by state"

### 3. value_counts — Frequency Distribution

Counts occurrences of each value. Examples: "How many customers per state?", "Distribution of contract types"

### 4. top_n — Rank and Filter

Top/bottom N records by column. Detects ascending/descending from keywords like "worst", "lowest".

### 5. cross_tab — Cross-Tabulation

Frequency table between two categorical columns. Examples: "Survival by class and sex"

### 6. smart_query — LLM-Generated Pandas (Fallback)

Last resort. LLM generates pandas code executed in sandboxed environment.

**Sandbox safety layers:**

1. **AST validation:** Max 10 lines, no imports, no `os`/`sys`/`eval`/`exec`/`open`, no dunders
2. **Safe proxies:** `pd` → `_SafePandas`, `np` → `_SafeNumpy`, restricted `__builtins__`
3. **Execution sandbox:** Daemon thread, 5s timeout with thread kill, circuit breaker (3 timeouts in 60s → disabled)

**Column auto-correction:** `_fix_column_refs()` fixes LLM typos via case-insensitive + fuzzy matching (cutoff=0.6).

### Column Name Resolution

All query skills use scored matching (`_resolve_column()`):

| Score | Match Type | Example |
|-------|-----------|---------|
| 100 | Exact | `"age"` → `"age"` |
| 95 | Normalized | `"monthly charges"` → `"monthly_charges"` |
| 80 | Word boundary | `"age"` → `"passenger_age"` (not `"passage"`) |
| 60 | Stem | `"survive"` → `"survived"` |
| 40 | Partial (guarded) | `"charge"` → `"monthly_charges"` (min 4 chars, max 3x length) |

### Routing Keywords

- `pivot_table`: "average X by Y", "sum of X per Y", "group by"
- `top_n`: "top N", "bottom N", "highest", "lowest", "rank"
- `query_data`: "filter", "where", "show me only"
- `value_counts`: "how many", "count of", "frequency"
- `cross_tab`: "cross tab", "contingency"
- `smart_query`: fallback when nothing else matches

---

## NLP Skills

5 NLP skills in `engine/datapilot/nlp/`. All degrade gracefully if optional dependencies are missing.

### analyze_sentiment (`nlp/sentiment.py`)

VADER (social media) + TextBlob (general). Column-level: runs on every row, returns % positive/negative/neutral.

VADER scoring: compound >= 0.05 → positive, <= -0.05 → negative, else neutral.

### extract_entities (`nlp/entities.py`)

spaCy NER (en_core_web_sm) with regex fallback. Entity types: PERSON, ORG, GPE, DATE, MONEY, PRODUCT, EVENT.

### extract_topics (`nlp/topics.py`)

LDA or NMF topic modeling with TF-IDF/Count vectorization. Parameters: text_column, n_topics (default: 5), method.

### detect_intent (`nlp/intent.py`)

TF-IDF + Logistic Regression for labeled data, keyword heuristics fallback. Categories: complaint, inquiry, feedback, request, praise, suggestion.

### analyze_text (`nlp/text_stats.py`)

Word count, sentence count, avg word length, vocabulary richness, TF-IDF keywords, top bigrams.

### NLP Dependencies

| Skill | Required | Optional |
|-------|----------|----------|
| sentiment | vaderSentiment | textblob |
| entities | — | spacy, en_core_web_sm |
| topics | scikit-learn | gensim |
| intent | scikit-learn, joblib | — |
| text_stats | — | nltk, textstat |

---

## Visualization

13 chart types in `engine/datapilot/viz/charts.py`. Static (matplotlib/seaborn at 150 DPI) + interactive (plotly HTML).

### Chart Types

| Type | Parameters | Best For |
|------|-----------|----------|
| `bar` | x, y, hue | Categorical comparisons |
| `barh` | x, y | Horizontal bars, long names |
| `line` | x, y | Trends over time |
| `scatter` | x, y, hue | Numeric relationships |
| `histogram` | x, bins (30) | Single numeric distribution |
| `box` | x, y | Distribution by category |
| `violin` | x, y | Distribution shape comparison |
| `heatmap` | (all numeric) | Correlation matrix |
| `pie` | x | Proportion (top 10) |
| `area` | x, y | Cumulative trends |
| `density` | x, hue | Smooth KDE curves |
| `count` | x, hue | Categorical frequency |
| `pair` | (first 6 numeric) | Pairwise relationships |

### Smart Type Inference

`_infer_chart_type()` picks the best chart when user doesn't specify:

| x type | y type | Result |
|--------|--------|--------|
| numeric | numeric | scatter |
| categorical | numeric | bar |
| numeric (alone) | — | histogram |
| categorical (alone) | — | count |

### Chart Summary (Anti-Hallucination)

`_compute_chart_summary()` generates structured data summaries passed to LLM for narration:

| Chart Type | Summary Includes |
|-----------|-----------------|
| bar (x+y) | Mean values per group, top 20 |
| scatter | Correlation coefficient, x/y stats |
| histogram | Mean, median, std, min, max |
| box | Median, Q1, Q3 per group |
| pie | Value counts with percentages |
| heatmap | Top 5 strongest correlation pairs |

### Auto-Dashboard

`create_dashboard()` generates multi-chart overview: histograms (2 numeric), count plots (2 categorical), scatter (first 2 numeric).

### LLM Chart Suggestions

`GET /api/chart/suggest` — Gemini primary, Groq fallback. Returns chart_type, x, y, title, reason. Column names must match actual dataset columns (spaces, not underscores).

---

## Export

3 report formats in `engine/datapilot/export/`. Endpoint: `POST /api/export/{format}`.

| Format | Library | Features |
|--------|---------|----------|
| **PDF** | ReportLab | Professional layout, numbered footers, embedded chart PNGs |
| **DOCX** | python-docx | Structured headings, auto page numbers, tables |
| **PPTX** | python-pptx | Title/content/closing slides, smart text splitting, word truncation |

Templates in `export/templates/`: Executive (summary-focused) and Detailed (full results).

---

## Testing

427 unit tests across 15 files, all passing. Uses pytest with mocked LLM providers.

### Running Tests

```bash
python -m pytest tests/ -v          # All tests
python -m pytest tests/ -q          # Quick run
python -m pytest tests/unit/test_query_skills.py -v   # Single file
python -m pytest tests/ -m "not slow"                 # Skip slow tests
```

### Test Organization

| File | Tests | Coverage |
|------|-------|----------|
| test_query_skills.py | 96 | All 6 query skills, sandbox safety, timeout |
| test_real_datasets.py | 91 | Every skill type on Superstore + Titanic |
| test_e2e_human_questions.py | 60 | Full pipeline: question → route → execute |
| test_semantic_router.py | 24 | Embedding skill matching, thresholds |
| test_fingerprint_d3.py | 23 | LLM understanding, DatasetUnderstanding |
| test_session_persistence.py | 23 | SQLite WAL, restore, expiry |
| test_failover.py | 19 | Task-aware routing (which provider) |
| test_fingerprint.py | 15 | Data snapshot, PII redaction |
| test_autopilot.py | 14 | AutoPilot plan + execution |
| test_session_store.py | 13 | Two-tier cache |
| test_router_correlation.py | 12 | Column matching, word boundary |
| test_production_flow.py | 12 | Full production flow |
| test_chart_suggest.py | 9 | Chart suggestions + inference |
| test_gemini_provider.py | 8 | Gemini API mocks |
| test_export_content.py | 8 | PDF/DOCX/PPTX export |

### Test Data

- **Sample - Superstore.csv** — Retail orders (Sales, Profit, Category, Region, etc.)
- **Titanic-Dataset.csv** — Passenger survival (Survived, Pclass, Sex, Age, Fare, etc.)

### Mocking Strategy

LLM providers are always mocked — never make real API calls in tests:

```python
mock_provider = MagicMock()
mock_provider.route_question.return_value = {
    "skill": "analyze_correlations",
    "parameters": {"target": "Churn"},
    "confidence": 0.9
}
```
