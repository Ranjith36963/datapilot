"""
Semantic skill routing using sentence-transformers embeddings.

Encodes skill descriptions and user questions into the same vector space,
then matches by cosine similarity. Handles paraphrased/synonym-heavy
questions that keyword regex would miss.

Model: all-MiniLM-L6-v2 (90MB, loads in ~2s, runs 100% locally).
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger("datapilot.core.semantic_router")

# ---------------------------------------------------------------------------
# Skill corpus — description + example phrasings per skill
# ---------------------------------------------------------------------------

SKILL_DESCRIPTIONS = {
    "pivot_table": (
        "Aggregate data by group. Average by category. Mean per group. "
        "Sum by region. Group by and calculate. Total per segment. "
        "Average fare by class. Mean salary by department. "
        "Break down values by category. "
        "Most profitable categories. Which category has the highest. "
        "Revenue by region. Total sales per product. Best performing groups. "
        "Compare totals across categories. Profit by segment."
    ),
    "top_n": (
        "Rank items by a column and return the top N or bottom N. "
        "Top 10 highest values. Bottom 5 lowest. Best performers ranked. "
        "Top earners. Highest salary. Cheapest items. Most expensive single item. "
        "Rank by score and show top 20."
    ),
    "query_data": (
        "Filter and select rows matching conditions. Show rows where column equals value. "
        "Filter by criteria. Select where equals. Show me only females. "
        "Display only males. Just the passengers below age 25. "
        "Records where status is active. Exclude rows. Include only. "
        "Show rows where price is greater than 100. Display rows where profit is negative. "
        "Show me only, display matching records, filter where."
    ),
    "value_counts": (
        "Count frequency of values in a column. How many per category. "
        "Frequency distribution. Count each value. Tally. "
        "How many males and females. Count by group. "
        "Number of occurrences. Distribution of categories."
    ),
    "cross_tab": (
        "Cross-tabulation of two categorical columns. Two-way frequency table. "
        "Contingency table. Breakdown by two categories simultaneously. "
        "Crosstab of column A vs column B. Two-variable frequency matrix. "
        "How are two categorical variables distributed together."
    ),
    "classify": (
        "Train a classification model to predict categories. "
        "Build a model to predict outcome. Build a predictive model. "
        "What factors determine outcome. Which features predict target. "
        "What drives survival. What causes churn. What predicts churn. "
        "Feature importance for prediction. Decision tree. Random forest. "
        "Predict category. Predict survival. Predict churn. "
        "Build classifier. Train classifier. Classification model. "
        "What contributes to, what influences, what affects the outcome. "
        "What factors predict. Which variables determine. "
        "Build a model to predict survival. Train a model to classify."
    ),
    "analyze_correlations": (
        "Find correlations between columns. Relationship between variables. "
        "How are X and Y related. How related are two columns. Association between features. "
        "How are age and fare related. Correlation matrix. Correlation analysis. "
        "Which variables are linked. Connection between columns. "
        "How does one variable relate to another. "
        "Correlate two columns. Is there an association between variables. "
        "How does X relate to Y. How is X related to Y. "
        "What is the relationship between columns. "
        "Show correlations between Age Fare and other numeric columns. "
        "Correlation between specific columns. How correlated are variables. "
        "Pearson correlation. Spearman correlation. "
        "Linear relationship between numeric variables."
    ),
    "detect_outliers": (
        "Find anomalies and outliers in the data. Unusual values. "
        "Extreme data points. Anomaly detection. Abnormal records. "
        "Suspicious entries. Find unusual values. Identify extremes. "
        "Data points that don't fit the pattern."
    ),
    "describe_data": (
        "Descriptive statistics and summary. Summary statistics of the data. "
        "Statistical overview. Mean median mode standard deviation. "
        "Basic stats. Describe columns. Distribution summary. "
        "What are the statistics. Data description. Column statistics."
    ),
    "find_clusters": (
        "Group similar records together. Segment customers. Cluster analysis. "
        "Find natural groups. K-means clustering. "
        "Grouping similar items. Identify segments. "
        "Partition data into groups based on similarity."
    ),
    "predict_numeric": (
        "Predict a numeric value using regression. Estimate a number. "
        "Predict price. Predict salary. Predict score. "
        "Numeric prediction. Linear regression. "
        "What will the value be. Estimate the amount."
    ),
    "forecast": (
        "Time series forecasting. Predict future values over time. "
        "Trend analysis. Seasonal forecast. Time-based prediction. "
        "Project forward. Predict future sales trend. "
        "What will happen next month. Future projection."
    ),
    "run_hypothesis_test": (
        "Statistical hypothesis testing. Is there a significant difference. "
        "Run a t-test. Run a chi-square test. ANOVA test. P-value. Statistical significance. "
        "Is the difference between groups significant. Is the difference statistically significant. "
        "Test whether two groups differ. Compare distributions statistically. "
        "T-test on column by group. Test if means differ between groups. "
        "Is there a significant difference in values between categories. "
        "Statistically significant difference between groups. "
        "Hypothesis test between two groups. Test for significance. "
        "Are the means significantly different. Is the difference real or random. "
        "Is there a significant difference in fare between survivors and non-survivors. "
        "Does salary differ significantly between departments. "
        "Test if price varies significantly across categories. "
        "Is the difference in age between males and females significant. "
        "Statistical test comparing two groups on a numeric variable."
    ),
    "analyze_sentiment": (
        "Sentiment analysis on text data. Positive or negative sentiment. "
        "Opinion mining. Emotion detection. Text mood analysis. "
        "Review sentiment. Feelings in text."
    ),
    "smart_query": (
        "Flexible data question answered with custom pandas code. "
        "Complex computation not matching other skills. "
        "Custom data transformation. Advanced calculation."
    ),
    "compare_groups": (
        "Compare metrics across groups. Difference between categories. "
        "Group comparison. Compare averages between segments. "
        "How do groups differ in values."
    ),
    "profile_data": (
        "Dataset overview and profiling. What is in this data. "
        "Data summary. Column types. Missing values. Data quality report. "
        "Tell me about the dataset. Overview of the data."
    ),
    "select_features": (
        "Feature selection and importance ranking. Which features matter most. "
        "Variable selection. Reduce features. Important columns. "
        "Which columns are most predictive."
    ),
    "reduce_dimensions": (
        "Dimensionality reduction. PCA principal component analysis. "
        "Reduce dimensions. Compress features. Visualization in 2D. "
        "Project high-dimensional data to lower dimensions."
    ),
    "create_chart": (
        "Create chart plot graph visualization. Draw a bar chart. "
        "Scatter plot of X vs Y. Histogram of distribution. Line chart trend. "
        "Box plot. Heatmap. Pie chart. Violin plot. Density plot. Pairplot. "
        "Visualize data. Show me a graph. Plot the relationship."
    ),
    "survival_analysis": (
        "Survival analysis. Kaplan-Meier curve. Hazard rate. "
        "Time to event. Survival probability. Censored data analysis. "
        "How long until event occurs. Duration analysis. "
        "Run a survival analysis on the data. Survival curve. "
        "Kaplan-Meier estimate. Cox proportional hazards. "
        "Event time analysis. Time-to-event modeling."
    ),
    "find_thresholds": (
        "Find optimal threshold cutoff split point. "
        "Best cutoff value for classification. Optimal split. "
        "Where to draw the line. Decision boundary."
    ),
    "explain_model": (
        "Explain model predictions. SHAP values. Feature contribution. "
        "Why did the model predict this. Model interpretability. "
        "What features contributed to the prediction."
    ),
    "calculate_effect_size": (
        "Effect size calculation. Cohen's d. Practical significance. "
        "How large is the difference. Magnitude of effect. "
        "Is the difference meaningful. Odds ratio. Cramer's V."
    ),
    "extract_topics": (
        "Topic modeling. Discover themes in text. Topic extraction. "
        "What topics are discussed. Theme analysis. "
        "Latent topics in documents."
    ),
    "extract_entities": (
        "Named entity recognition NER. Extract entities from text. "
        "Find names places organizations in text. Entity extraction. "
        "Identify people locations companies."
    ),
    "validate_data": (
        "Data quality validation. Check data integrity. "
        "Find data quality issues. Missing values check. "
        "Validate data types. Data cleaning assessment."
    ),
    "engineer_features": (
        "Feature engineering. Create new features from existing columns. "
        "Auto-generate features. Date features. Interaction features. "
        "Binning numeric columns. Encode categorical variables. "
        "Transform and enrich the data."
    ),
    "analyze_time_series": (
        "Time series analysis. Decompose trend seasonality residuals. "
        "Stationarity test. Autocorrelation. Time series components. "
        "Analyze temporal patterns. Time series decomposition."
    ),
    "detect_change_points": (
        "Detect change points in time series. Structural breaks. "
        "Where did the trend change. Regime change detection. "
        "Find shifts in the data over time."
    ),
    "analyze_text": (
        "Text statistics and analysis. Word count. Readability score. "
        "Text length. Vocabulary richness. Text complexity. "
        "Analyze writing style. Text summary statistics."
    ),
    "detect_intent": (
        "Classify text intent. What is the purpose of the message. "
        "Intent classification. Categorize user messages. "
        "Is this a question complaint or request."
    ),
    "fingerprint_dataset": (
        "Dataset fingerprinting. What kind of data is this. "
        "Identify the domain of the dataset. Dataset characterization. "
        "What type of dataset is this. Data domain detection."
    ),
}


class SemanticSkillMatcher:
    """Embedding-based skill routing using sentence-transformers.

    Singleton pattern — model loaded once, cached forever after.
    """

    _instance: Optional["SemanticSkillMatcher"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._cos_sim = _cos_sim

            # Build corpus
            self.skill_names = list(SKILL_DESCRIPTIONS.keys())
            corpus_texts = [SKILL_DESCRIPTIONS[s] for s in self.skill_names]
            self.corpus_embeddings = self.model.encode(corpus_texts)

            self._initialized = True
            logger.info(
                f"SemanticSkillMatcher loaded: {len(self.skill_names)} skills, "
                f"embedding dim={self.corpus_embeddings.shape[1]}"
            )
        except Exception as e:
            logger.warning(f"SemanticSkillMatcher init failed: {e}")
            self._initialized = False
            raise

    def match(
        self,
        question: str,
        threshold: float = 0.35,
    ) -> Optional[Tuple[str, float]]:
        """Match a question to the best skill by cosine similarity.

        Returns (skill_name, score) if score >= threshold, else None.
        """
        if not self._initialized:
            return None

        q_embedding = self.model.encode([question])
        similarities = self._cos_sim(q_embedding, self.corpus_embeddings)
        best_idx = int(similarities.argmax())
        best_score = float(similarities[0][best_idx])

        logger.debug(
            f"Semantic match: '{question[:50]}...' -> "
            f"{self.skill_names[best_idx]} ({best_score:.3f})"
        )

        if best_score >= threshold:
            return (self.skill_names[best_idx], best_score)
        return None


def get_semantic_matcher() -> Optional[SemanticSkillMatcher]:
    """Get the singleton SemanticSkillMatcher, or None if unavailable."""
    try:
        return SemanticSkillMatcher()
    except Exception:
        return None
