"""
Veritly AI - Market Intelligence Data Platform
Main orchestration script for the data pipeline.

IMPORTANT: This pipeline uses REAL sklearn machine learning.
All model coefficients are LEARNED from data, not hardcoded.

Usage:
    python main.py                    # Run full pipeline (trains model)
    python main.py --step ingest      # Run only ingestion
    python main.py --step curate      # Run only curation
    python main.py --step analyze     # Run only analysis
    python main.py --step report      # Generate report only
    python main.py --visualise        # Generate visualisation charts
    python main.py --export pdf       # Export to PDF
    python main.py --export docx      # Export to Word
    python main.py --export pptx      # Export to PowerPoint
    python main.py --export all       # Export to all formats
    python main.py --load-model       # Use existing trained model instead of retraining
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASE_DIR, RAW_ZONE, CURATED_ZONE, DERIVED_ZONE, REPORTS_DIR
from src.utils import setup_logging, MetadataManager
from src.ingestion.file_loader import FileLoader
from src.processing.curation import DataCurator
from src.processing.derivation import DataDeriver
from src.analysis.analyzer import ChurnAnalyzer
from src.ai.ai_analyst import AIAnalyst
from src.products.report_generator import ReportGenerator
from src.products.visualisation import ChurnVisualiser
from src.products.export import ReportExporter


def run_pipeline(
    source_file: str = "Raw Data.xlsx",
    version: str = "v1",
    steps: list = None,
    visualise: bool = False,
    export_format: str = None,
    load_existing_model: bool = False
):
    """
    Run the complete Veritly data pipeline.

    This pipeline uses REAL sklearn machine learning:
    - StandardScaler.fit() CALCULATES mean/std from data
    - LogisticRegression.fit() LEARNS coefficients from data
    - sklearn.metrics CALCULATES accuracy, precision, recall, etc.

    NO hardcoded coefficients. Everything is learned from input data.
    """
    logger = setup_logging("main")
    logger.info("=" * 70)
    logger.info("VERITLY AI - Market Intelligence Data Platform")
    logger.info("REAL MACHINE LEARNING - All coefficients LEARNED from data")
    logger.info("=" * 70)

    all_steps = ["ingest", "curate", "analyze", "derive", "report"]
    steps_to_run = steps if steps else all_steps

    outputs = {
        "raw_path": None, "raw_df": None,
        "curated_path": None, "curated_df": None,
        "analysis_results": None, "ai_insights": None,
        "training_results": None,  # NEW: Store sklearn training results
        "derived_paths": [], "report_path": None,
        "visualisation_paths": {}, "export_paths": {}
    }

    # STEP 1: INGESTION
    if "ingest" in steps_to_run:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 40)

        source_path = BASE_DIR / source_file
        if not source_path.exists():
            logger.error(f"Source file not found: {source_path}")
            return outputs

        loader = FileLoader()
        raw_path, raw_df = loader.ingest(
            source_path=source_path,
            dataset_name="raw_data",
            description="Telecom customer data for churn analysis"
        )
        outputs["raw_path"] = raw_path
        outputs["raw_df"] = raw_df
        logger.info(f"Ingested {len(raw_df)} rows to {raw_path}")

    # STEP 2: CURATION
    if "curate" in steps_to_run:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: DATA CURATION")
        logger.info("=" * 40)

        if outputs["raw_df"] is None:
            loader = FileLoader()
            raw_files = list(RAW_ZONE.glob("*.xlsx"))
            if raw_files:
                outputs["raw_df"] = loader.load_existing(raw_files[0].name)
            else:
                logger.error("No raw data found. Run ingestion first.")
                return outputs

        curator = DataCurator()
        curated_df, curated_path = curator.curate(
            df=outputs["raw_df"],
            source_dataset_id="raw_raw_data",
            output_name="telecom_customers",
            version=version
        )
        outputs["curated_df"] = curated_df
        outputs["curated_path"] = curated_path
        logger.info(f"Curated {len(curated_df)} rows to {curated_path}")

    # STEP 3: ANALYSIS (basic stats, done before ML training)
    if "analyze" in steps_to_run:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: CHURN ANALYSIS (Statistical)")
        logger.info("=" * 40)

        if outputs["curated_df"] is None:
            import pandas as pd
            curated_files = list(CURATED_ZONE.glob("*.xlsx"))
            if curated_files:
                outputs["curated_df"] = pd.read_excel(curated_files[0], engine="openpyxl")
            else:
                logger.error("No curated data found. Run curation first.")
                return outputs

        # Run statistical analysis (segments, tipping points, etc.)
        # Model metrics will be added after STEP 4 training
        analyzer = ChurnAnalyzer()
        # Don't pass training_results yet - will update after derivation
        analysis_results = analyzer.analyze(outputs["curated_df"])
        outputs["analysis_results"] = analysis_results

        summary = analyzer.get_summary()
        logger.info(f"\nAnalysis Summary:")
        logger.info(f"  Total Customers: {summary['total_customers']}")
        logger.info(f"  Churn Rate: {summary['churn_rate']}%")
        logger.info(f"  Churned Count: {summary['churned_count']}")

    # STEP 4: DERIVATION (includes sklearn ML training)
    if "derive" in steps_to_run:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: DATA DERIVATION (sklearn ML Training)")
        logger.info("=" * 40)
        logger.info("")
        logger.info("This step TRAINS a logistic regression model using sklearn.")
        logger.info("All coefficients are LEARNED from data, NOT hardcoded.")
        logger.info("")

        if outputs["curated_df"] is None:
            import pandas as pd
            curated_files = list(CURATED_ZONE.glob("*.xlsx"))
            if curated_files:
                outputs["curated_df"] = pd.read_excel(curated_files[0], engine="openpyxl")
            else:
                logger.error("Curated data missing. Run previous steps.")
                return outputs

        deriver = DataDeriver()

        # Train model and create risk scores
        # force_retrain=True means always train new model
        # force_retrain=False (--load-model) means use existing if available
        risk_df, risk_path = deriver.create_risk_scores(
            df=outputs["curated_df"],
            source_dataset_id=f"curated_telecom_customers_{version}",
            version=version,
            force_retrain=not load_existing_model
        )
        outputs["derived_paths"].append(risk_path)
        outputs["training_results"] = deriver.get_training_results()

        # Log the LEARNED model info
        logger.info("")
        logger.info("=" * 50)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("=" * 50)

        tr = outputs["training_results"]
        logger.info(f"Accuracy:  {tr['accuracy']*100:.2f}% (CALCULATED by sklearn)")
        logger.info(f"Precision: {tr['precision']*100:.2f}% (CALCULATED by sklearn)")
        logger.info(f"Recall:    {tr['recall']*100:.2f}% (CALCULATED by sklearn)")
        logger.info(f"High-risk: {tr['high_risk_count']} customers")

        # Now run analysis again with training results
        if outputs["analysis_results"]:
            analyzer = ChurnAnalyzer()
            outputs["analysis_results"] = analyzer.analyze(
                outputs["curated_df"],
                training_results=outputs["training_results"]
            )

        # Create churn analysis summary
        if outputs["analysis_results"]:
            analysis_df, analysis_path = deriver.create_churn_analysis(
                df=outputs["curated_df"],
                analysis_results=outputs["analysis_results"],
                source_dataset_id=f"curated_telecom_customers_{version}",
                version=version
            )
            outputs["derived_paths"].append(analysis_path)

        # Generate AI insights
        logger.info("\nGenerating AI Insights...")
        ai_analyst = AIAnalyst()
        ai_insights = ai_analyst.analyze_churn_patterns(
            outputs["curated_df"],
            outputs["analysis_results"]
        )
        outputs["ai_insights"] = ai_insights
        logger.info(f"  Generated {len(ai_insights.get('insights', []))} insights")

    # STEP 5: REPORTING
    if "report" in steps_to_run:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 5: REPORT GENERATION")
        logger.info("=" * 40)

        if outputs["analysis_results"] is None or outputs["ai_insights"] is None:
            logger.error("Analysis results missing. Run analysis first.")
            return outputs

        report_gen = ReportGenerator()
        report_path = report_gen.generate_executive_report(
            analysis_results=outputs["analysis_results"],
            ai_insights=outputs["ai_insights"],
            df=outputs["curated_df"],
            version=version
        )
        outputs["report_path"] = report_path
        logger.info(f"Generated executive report: {report_path}")

    # STEP 6: VISUALISATION
    if visualise or export_format:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 6: VISUALISATION")
        logger.info("=" * 40)

        if outputs["curated_df"] is None or outputs["analysis_results"] is None:
            import pandas as pd
            curated_files = list(CURATED_ZONE.glob("*.xlsx"))
            if curated_files:
                outputs["curated_df"] = pd.read_excel(curated_files[0], engine="openpyxl")

                # Re-run analysis if needed
                if outputs["analysis_results"] is None:
                    analyzer = ChurnAnalyzer()
                    outputs["analysis_results"] = analyzer.analyze(outputs["curated_df"])

                    ai_analyst = AIAnalyst()
                    outputs["ai_insights"] = ai_analyst.analyze_churn_patterns(
                        outputs["curated_df"], outputs["analysis_results"]
                    )
            else:
                logger.error("No curated data found. Run curation first.")
                return outputs

        visualiser = ChurnVisualiser()
        vis_paths = visualiser.generate_all(
            df=outputs["curated_df"],
            analysis_results=outputs["analysis_results"],
            version=version
        )
        outputs["visualisation_paths"] = vis_paths

        for name, path in vis_paths.items():
            logger.info(f"  Generated: {path}")

    # STEP 7: EXPORT
    if export_format:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 7: EXPORT")
        logger.info("=" * 40)

        if not outputs["visualisation_paths"]:
            logger.error("Visualisations required for export. Run with --visualise first.")
            return outputs

        exporter = ReportExporter()

        if export_format == "all":
            export_paths = exporter.export_all(
                analysis_results=outputs["analysis_results"],
                ai_insights=outputs["ai_insights"],
                df=outputs["curated_df"],
                visualisation_paths=outputs["visualisation_paths"],
                version=version
            )
            outputs["export_paths"] = export_paths
        elif export_format == "pdf":
            outputs["export_paths"]["pdf"] = exporter.export_to_pdf(
                outputs["analysis_results"], outputs["ai_insights"],
                outputs["curated_df"], outputs["visualisation_paths"], version
            )
        elif export_format == "docx":
            outputs["export_paths"]["docx"] = exporter.export_to_docx(
                outputs["analysis_results"], outputs["ai_insights"],
                outputs["curated_df"], outputs["visualisation_paths"], version
            )
        elif export_format == "pptx":
            outputs["export_paths"]["pptx"] = exporter.export_to_pptx(
                outputs["analysis_results"], outputs["ai_insights"],
                outputs["curated_df"], outputs["visualisation_paths"], version
            )

        for fmt, path in outputs["export_paths"].items():
            logger.info(f"  Exported {fmt.upper()}: {path}")

    # SUMMARY
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    logger.info("\nOutputs Generated:")
    if outputs["raw_path"]:
        logger.info(f"  Raw Data: {outputs['raw_path']}")
    if outputs["curated_path"]:
        logger.info(f"  Curated Data: {outputs['curated_path']}")
    for path in outputs["derived_paths"]:
        logger.info(f"  Derived Data: {path}")
    if outputs["report_path"]:
        logger.info(f"  Executive Report: {outputs['report_path']}")
    for name, path in outputs.get("visualisation_paths", {}).items():
        logger.info(f"  Visualisation: {path}")
    for fmt, path in outputs.get("export_paths", {}).items():
        logger.info(f"  Export ({fmt.upper()}): {path}")

    # Show training results summary
    if outputs["training_results"]:
        tr = outputs["training_results"]
        logger.info("\nModel Training Results (sklearn - CALCULATED from data):")
        logger.info(f"  Accuracy:  {tr['accuracy']*100:.2f}%")
        logger.info(f"  Precision: {tr['precision']*100:.2f}%")
        logger.info(f"  Recall:    {tr['recall']*100:.2f}%")
        logger.info(f"  F1 Score:  {tr['f1_score']:.4f}")
        logger.info(f"  High-risk customers: {tr['high_risk_count']}")
        logger.info(f"  Model saved to: models/churn_model.pkl")

    if outputs["analysis_results"]:
        logger.info("\nKey Findings:")
        logger.info(f"  Churn Rate: {outputs['analysis_results']['overall_churn_rate']}%")
        for tp in outputs["analysis_results"].get("tipping_points", []):
            logger.info(f"  {tp['factor']}: {tp['churn_above']}% churn above threshold ({tp['threshold']})")

    return outputs


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Veritly AI - Market Intelligence Data Platform (REAL ML)"
    )
    parser.add_argument("--source", default="Raw Data.xlsx",
                       help="Source data file name")
    parser.add_argument("--version", default="v1",
                       help="Version string for outputs")
    parser.add_argument("--step", choices=["ingest", "curate", "analyze", "derive", "report"],
                       help="Run only a specific step")
    parser.add_argument("--visualise", action="store_true",
                       help="Generate visualisation charts")
    parser.add_argument("--export", choices=["pdf", "docx", "pptx", "all"],
                       help="Export report to specified format(s)")
    parser.add_argument("--load-model", action="store_true",
                       help="Load existing trained model instead of retraining")

    args = parser.parse_args()
    steps = [args.step] if args.step else None

    run_pipeline(
        source_file=args.source,
        version=args.version,
        steps=steps,
        visualise=args.visualise,
        export_format=args.export,
        load_existing_model=args.load_model
    )


if __name__ == "__main__":
    main()
