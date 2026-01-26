"""
Veritly AI - Market Intelligence Data Platform
Main orchestration script for the data pipeline.

Usage:
    python main.py                    # Run full pipeline
    python main.py --step ingest      # Run only ingestion
    python main.py --step curate      # Run only curation
    python main.py --step analyze     # Run only analysis
    python main.py --step report      # Generate report only
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


def run_pipeline(source_file: str = "Raw Data.xlsx", version: str = "v1", steps: list = None):
    """Run the complete Veritly data pipeline."""
    logger = setup_logging("main")
    logger.info("=" * 60)
    logger.info("VERITLY AI - Market Intelligence Data Platform")
    logger.info("=" * 60)

    all_steps = ["ingest", "curate", "analyze", "derive", "report"]
    steps_to_run = steps if steps else all_steps

    outputs = {
        "raw_path": None, "raw_df": None,
        "curated_path": None, "curated_df": None,
        "analysis_results": None, "ai_insights": None,
        "derived_paths": [], "report_path": None
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

    # STEP 3: ANALYSIS
    if "analyze" in steps_to_run:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: CHURN ANALYSIS")
        logger.info("=" * 40)

        if outputs["curated_df"] is None:
            import pandas as pd
            curated_files = list(CURATED_ZONE.glob("*.xlsx"))
            if curated_files:
                outputs["curated_df"] = pd.read_excel(curated_files[0], engine="openpyxl")
            else:
                logger.error("No curated data found. Run curation first.")
                return outputs

        analyzer = ChurnAnalyzer()
        analysis_results = analyzer.analyze(outputs["curated_df"])
        outputs["analysis_results"] = analysis_results

        summary = analyzer.get_summary()
        logger.info(f"\nAnalysis Summary:")
        logger.info(f"  Total Customers: {summary['total_customers']}")
        logger.info(f"  Churn Rate: {summary['churn_rate']}%")
        logger.info(f"  Churned Count: {summary['churned_count']}")

        logger.info("\nGenerating AI Insights...")
        ai_analyst = AIAnalyst()
        ai_insights = ai_analyst.analyze_churn_patterns(outputs["curated_df"], analysis_results)
        outputs["ai_insights"] = ai_insights
        logger.info(f"  Generated {len(ai_insights.get('insights', []))} insights")

    # STEP 4: DERIVATION
    if "derive" in steps_to_run:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: DATA DERIVATION")
        logger.info("=" * 40)

        if outputs["curated_df"] is None or outputs["analysis_results"] is None:
            logger.error("Curated data or analysis results missing. Run previous steps.")
            return outputs

        deriver = DataDeriver()

        risk_df, risk_path = deriver.create_risk_scores(
            df=outputs["curated_df"],
            source_dataset_id=f"curated_telecom_customers_{version}",
            version=version
        )
        outputs["derived_paths"].append(risk_path)
        logger.info(f"Created risk scores: {risk_path}")

        analysis_df, analysis_path = deriver.create_churn_analysis(
            df=outputs["curated_df"],
            analysis_results=outputs["analysis_results"],
            source_dataset_id=f"curated_telecom_customers_{version}",
            version=version
        )
        outputs["derived_paths"].append(analysis_path)
        logger.info(f"Created churn analysis: {analysis_path}")

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

    # SUMMARY
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    logger.info("\nOutputs Generated:")
    if outputs["raw_path"]:
        logger.info(f"  Raw Data: {outputs['raw_path']}")
    if outputs["curated_path"]:
        logger.info(f"  Curated Data: {outputs['curated_path']}")
    for path in outputs["derived_paths"]:
        logger.info(f"  Derived Data: {path}")
    if outputs["report_path"]:
        logger.info(f"  Executive Report: {outputs['report_path']}")

    if outputs["analysis_results"]:
        logger.info("\nKey Findings:")
        logger.info(f"  Churn Rate: {outputs['analysis_results']['overall_churn_rate']}%")
        for tp in outputs["analysis_results"].get("tipping_points", []):
            logger.info(f"  {tp['factor']}: {tp['churn_above']}% churn above threshold ({tp['threshold']})")

    return outputs


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Veritly AI - Market Intelligence Data Platform")
    parser.add_argument("--source", default="Raw Data.xlsx", help="Source data file name")
    parser.add_argument("--version", default="v1", help="Version string for outputs")
    parser.add_argument("--step", choices=["ingest", "curate", "analyze", "derive", "report"], help="Run only a specific step")

    args = parser.parse_args()
    steps = [args.step] if args.step else None

    run_pipeline(source_file=args.source, version=args.version, steps=steps)


if __name__ == "__main__":
    main()
