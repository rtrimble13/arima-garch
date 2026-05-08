"""
Command-line interface for ag-viz.

Provides Click-based CLI commands for visualizing ARIMA-GARCH model outputs.
"""

import sys
from pathlib import Path
from typing import Optional
import click

from ag_viz.utils import run_ag_command, ensure_output_dir, find_ag_executable
from ag_viz.io import load_csv_data, load_model_json, load_forecast_csv, load_diagnostics_json, parse_simulation_csv
from ag_viz.plotting import (
    plot_fit_diagnostics,
    plot_forecast,
    plot_residual_diagnostics,
    plot_simulation_paths,
)
from ag_viz.markdown_reports import (
    generate_fit_report,
    generate_forecast_report,
    generate_diagnostics_report,
    generate_simulation_report,
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    ag-viz: Visualization tools for ARIMA-GARCH models.

    This tool wraps the C++ 'ag' CLI and generates publication-quality
    plots for model outputs.
    """
    ag_path = find_ag_executable()
    if ag_path is None:
        click.echo(
            "Warning: ag executable not found. Please build the project or set "
            "the AG_EXECUTABLE environment variable.",
            err=True,
        )


# ---------------------------------------------------------------------------
# Primary workflow commands (run ag + generate outputs)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("-d", "--data", "data_path", required=True, type=click.Path(exists=True),
              help="Input data file in CSV format")
@click.option("-a", "--arima", required=True,
              help="ARIMA order as p,d,q (e.g., 1,0,1)")
@click.option("-g", "--garch", required=True,
              help="GARCH order as p,q (e.g., 1,1)")
@click.option("-o", "--output", "output_path", type=click.Path(),
              help="Output model file in JSON format")
@click.option("--plot", "plot_path", type=click.Path(),
              help="Full path to save diagnostic plot (overrides --plot-dir)")
@click.option("--plot-dir", type=click.Path(), default="./output",
              help="Directory to save diagnostic plot (default: ./output)")
@click.option("--show", is_flag=True, help="Display the plot")
@click.option("--markdown", is_flag=True,
              help="Generate a professional Markdown report with analysis and visuals")
@click.option("--report-dir", type=click.Path(), default="./reports",
              help="Directory to save the Markdown report (default: ./reports)")
def fit(data_path: str, arima: str, garch: str, output_path: Optional[str],
        plot_path: Optional[str], plot_dir: str, show: bool, markdown: bool,
        report_dir: str):
    """
    Fit ARIMA-GARCH model and generate diagnostic plots.

    Examples:
        ag-viz fit -d data.csv -a 1,0,1 -g 1,1 -o model.json
        ag-viz fit -d data.csv -a 1,0,1 -g 1,1 -o model.json --markdown
        ag-viz fit -d data.csv -a 1,0,1 -g 1,1 -o model.json --show
    """
    click.echo(f"Fitting model: ARIMA({arima})-GARCH({garch})...")

    if output_path is None:
        output_path = "model.json"

    try:
        args = ["fit", "-d", data_path, "-a", arima, "-g", garch, "-o", output_path]
        result = run_ag_command(args)
        click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)

        click.echo(f"\nGenerating diagnostic plots...")
        data = load_csv_data(Path(data_path))
        model = load_model_json(Path(output_path))

        if plot_path:
            p = Path(plot_path)
            actual_plot = plot_fit_diagnostics(
                data, model, p.parent, show=show, output_filename=p.name
            )
        else:
            actual_plot = plot_fit_diagnostics(
                data, model, Path(plot_dir), show=show
            )
        click.echo(f"✓ Saved fit diagnostics to: {actual_plot}")

        if markdown:
            report_path = Path(report_dir) / "fit_report.md"
            click.echo("\nGenerating Markdown report...")
            report_file = generate_fit_report(
                data=data,
                model_json=model,
                plot_path=actual_plot,
                output_path=report_path,
                use_data_uri=False,
            )
            click.echo(f"✓ Saved Markdown report to: {report_file}")

        click.echo(f"\n✓ Model saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Input model file in JSON format")
@click.option("-n", "--horizon", default=10, type=int,
              help="Forecast horizon (number of steps ahead)")
@click.option("-o", "--output", "output_path", type=click.Path(),
              help="Output forecast file in CSV format")
@click.option("--plot", "plot_path", type=click.Path(),
              help="Path to save forecast plot")
@click.option("--show", is_flag=True, help="Display the plot")
@click.option("--markdown", is_flag=True,
              help="Generate a professional Markdown report with analysis and visuals")
@click.option("--report-dir", type=click.Path(), default="./reports",
              help="Directory to save the Markdown report (default: ./reports)")
def forecast(model_path: str, horizon: int, output_path: Optional[str],
             plot_path: Optional[str], show: bool, markdown: bool, report_dir: str):
    """
    Generate forecasts and plot with confidence intervals.

    Examples:
        ag-viz forecast -m model.json -n 30 -o forecast.csv
        ag-viz forecast -m model.json -n 30 -o forecast.csv --markdown
    """
    click.echo(f"Generating {horizon}-step forecast...")

    if output_path is None:
        output_path = "forecast.csv"

    try:
        args = ["forecast", "-m", model_path, "-n", str(horizon), "-o", output_path]
        result = run_ag_command(args)
        click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)

        click.echo("\nGenerating forecast plot...")
        save_path = plot_forecast(
            Path(model_path),
            Path(output_path),
            confidence_levels=[0.68, 0.95],
            show=show,
            save=Path(plot_path) if plot_path else None,
        )
        click.echo(f"✓ Saved forecast plot to: {save_path}")

        if markdown:
            report_path = Path(report_dir) / "forecast_report.md"
            click.echo("\nGenerating Markdown report...")
            model = load_model_json(Path(model_path))
            forecast_data = load_forecast_csv(Path(output_path))
            report_file = generate_forecast_report(
                model_json=model,
                forecast_df=forecast_data,
                plot_path=save_path,
                output_path=report_path,
                use_data_uri=False,
            )
            click.echo(f"✓ Saved Markdown report to: {report_file}")

        click.echo(f"✓ Forecast saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Input model file in JSON format")
@click.option("-d", "--data", "data_path", required=True, type=click.Path(exists=True),
              help="Input data file in CSV format")
@click.option("-o", "--output", "output_dir", type=click.Path(), default="./diagnostics",
              help="Output directory for diagnostic plots and JSON")
@click.option("--show", is_flag=True, help="Display the plot")
@click.option("--markdown", is_flag=True,
              help="Generate a professional Markdown report with analysis and visuals")
@click.option("--report-dir", type=click.Path(), default="./reports",
              help="Directory to save the Markdown report (default: ./reports)")
def diagnostics(model_path: str, data_path: str, output_dir: str, show: bool,
                markdown: bool, report_dir: str):
    """
    Generate comprehensive residual diagnostic plots.

    Examples:
        ag-viz diagnostics -m model.json -d data.csv -o ./diagnostics/
        ag-viz diagnostics -m model.json -d data.csv -o ./diagnostics/ --markdown --show
    """
    click.echo("Running diagnostics...")

    output_path = Path(output_dir)
    ensure_output_dir(output_path)

    try:
        diag_json = output_path / "diagnostics.json"
        args = ["diagnostics", "-m", model_path, "-d", data_path, "-o", str(diag_json)]
        result = run_ag_command(args)
        click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)

        click.echo(f"\nGenerating diagnostic plots in {output_dir}...")
        plot_path = plot_residual_diagnostics(
            Path(model_path),
            Path(data_path),
            diag_json if diag_json.exists() else None,
            output_path,
            show=show,
        )
        click.echo(f"✓ Saved residual diagnostics to: {plot_path}")

        if markdown:
            report_path = Path(report_dir) / "diagnostics_report.md"
            click.echo("\nGenerating Markdown report...")
            model = load_model_json(Path(model_path))
            data = load_csv_data(Path(data_path))
            diagnostics_data = load_diagnostics_json(diag_json) if diag_json.exists() else None
            report_file = generate_diagnostics_report(
                model_json=model,
                data=data,
                diagnostics_json=diagnostics_data,
                plot_path=plot_path,
                output_path=report_path,
                use_data_uri=False,
            )
            click.echo(f"✓ Saved Markdown report to: {report_file}")

        click.echo(f"✓ Diagnostics saved to: {diag_json}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Input model file in JSON format")
@click.option("-p", "--paths", default=100, type=int,
              help="Number of simulation paths to generate")
@click.option("-n", "--length", default=1000, type=int,
              help="Number of observations per path")
@click.option("-s", "--seed", default=42, type=int,
              help="Random seed for reproducibility")
@click.option("-o", "--output", "output_path", type=click.Path(),
              help="Output simulation file in CSV format")
@click.option("--plot", "plot_path", type=click.Path(),
              help="Path to save simulation plot")
@click.option("--n-plot", default=10, type=int,
              help="Number of paths to plot")
@click.option("--show", is_flag=True, help="Display the plot")
@click.option("--stats", is_flag=True, help="Compute and display summary statistics")
@click.option("--markdown", is_flag=True,
              help="Generate a professional Markdown report with analysis and visuals")
@click.option("--report-dir", type=click.Path(), default="./reports",
              help="Directory to save the Markdown report (default: ./reports)")
def simulate(model_path: str, paths: int, length: int, seed: int,
             output_path: Optional[str], plot_path: Optional[str],
             n_plot: int, show: bool, stats: bool, markdown: bool, report_dir: str):
    """
    Simulate paths and visualize distributions.

    Examples:
        ag-viz simulate -m model.json -p 100 -n 1000 -o simulation.csv
        ag-viz simulate -m model.json -p 100 -n 1000 -o simulation.csv --markdown
    """
    click.echo(f"Simulating {paths} paths with {length} observations each...")

    if output_path is None:
        output_path = "simulation.csv"

    try:
        args = [
            "simulate",
            "-m", model_path,
            "-p", str(paths),
            "-n", str(length),
            "-s", str(seed),
            "-o", output_path,
        ]
        if stats:
            args.append("--stats")

        result = run_ag_command(args)
        click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)

        click.echo("\nGenerating simulation plot...")
        save_path = plot_simulation_paths(
            Path(output_path),
            n_paths_to_plot=n_plot,
            output_path=Path(plot_path) if plot_path else None,
            show=show,
        )
        click.echo(f"✓ Saved simulation plot to: {save_path}")

        if markdown:
            report_path = Path(report_dir) / "simulation_report.md"
            click.echo("\nGenerating Markdown report...")
            model = load_model_json(Path(model_path))
            simulation_data, _, _ = parse_simulation_csv(Path(output_path))
            report_file = generate_simulation_report(
                model_json=model,
                simulation_df=simulation_data,
                plot_path=save_path,
                output_path=report_path,
                n_paths=paths,
                length=length,
                use_data_uri=False,
            )
            click.echo(f"✓ Saved Markdown report to: {report_file}")

        click.echo(f"✓ Simulation data saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 'plot' subgroup — generate plots from existing output files, no ag subprocess
# ---------------------------------------------------------------------------

@cli.group()
def plot():
    """Generate plots from existing output files without re-running ag."""
    pass


@plot.command("fit")
@click.option("-d", "--data", "data_path", required=True, type=click.Path(exists=True),
              help="Input data CSV file")
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Fitted model JSON file")
@click.option("-o", "--output", "output_path", type=click.Path(),
              default="./output/fit_diagnostics.png",
              help="Path to save the plot (default: ./output/fit_diagnostics.png)")
@click.option("--show", is_flag=True, help="Display the plot")
def plot_fit(data_path: str, model_path: str, output_path: str, show: bool):
    """Plot fit diagnostics from an existing model file."""
    try:
        p = Path(output_path)
        data = load_csv_data(Path(data_path))
        model = load_model_json(Path(model_path))
        saved = plot_fit_diagnostics(data, model, p.parent, show=show, output_filename=p.name)
        click.echo(f"✓ Saved fit diagnostics to: {saved}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@plot.command("forecast")
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Fitted model JSON file")
@click.option("-f", "--forecast", "forecast_path", required=True, type=click.Path(exists=True),
              help="Forecast CSV file")
@click.option("-o", "--output", "output_path", type=click.Path(), default="forecast.png",
              help="Path to save the plot (default: forecast.png)")
@click.option("--show", is_flag=True, help="Display the plot")
@click.option("--confidence-levels", "confidence_levels", default="0.68,0.95",
              help="Comma-separated confidence levels (default: 0.68,0.95)")
def plot_forecast_cmd(model_path: str, forecast_path: str, output_path: str,
                      show: bool, confidence_levels: str):
    """Plot forecast with confidence intervals from existing output files."""
    try:
        levels = [float(x.strip()) for x in confidence_levels.split(",")]
        saved = plot_forecast(
            Path(model_path),
            Path(forecast_path),
            confidence_levels=levels,
            show=show,
            save=Path(output_path),
        )
        click.echo(f"✓ Saved forecast plot to: {saved}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@plot.command("diagnostics")
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Fitted model JSON file")
@click.option("-d", "--data", "data_path", required=True, type=click.Path(exists=True),
              help="Input data CSV file")
@click.option("-j", "--diagnostics-json", "diag_json_path", type=click.Path(exists=True),
              help="Diagnostics JSON file (optional)")
@click.option("-o", "--output", "output_path", type=click.Path(),
              default="./diagnostics/residual_diagnostics.png",
              help="Path to save the plot")
@click.option("--show", is_flag=True, help="Display the plot")
def plot_diagnostics(model_path: str, data_path: str, diag_json_path: Optional[str],
                     output_path: str, show: bool):
    """Plot residual diagnostics from existing output files."""
    try:
        p = Path(output_path)
        saved = plot_residual_diagnostics(
            Path(model_path),
            Path(data_path),
            Path(diag_json_path) if diag_json_path else None,
            p.parent,
            show=show,
            output_filename=p.name,
        )
        click.echo(f"✓ Saved residual diagnostics to: {saved}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@plot.command("simulate")
@click.option("-s", "--simulation", "sim_path", required=True, type=click.Path(exists=True),
              help="Simulation CSV file")
@click.option("--n-plot", default=10, type=int,
              help="Number of individual paths to display (default: 10)")
@click.option("-o", "--output", "output_path", type=click.Path(),
              default="simulation_paths.png",
              help="Path to save the plot (default: simulation_paths.png)")
@click.option("--show", is_flag=True, help="Display the plot")
def plot_simulate(sim_path: str, n_plot: int, output_path: str, show: bool):
    """Plot simulation paths from an existing simulation CSV."""
    try:
        saved = plot_simulation_paths(
            Path(sim_path),
            n_paths_to_plot=n_plot,
            output_path=Path(output_path),
            show=show,
        )
        click.echo(f"✓ Saved simulation plot to: {saved}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 'report' subgroup — generate Markdown reports from existing output files
# ---------------------------------------------------------------------------

@cli.group()
def report():
    """Generate Markdown reports from existing output files without re-running ag."""
    pass


@report.command("fit")
@click.option("-d", "--data", "data_path", required=True, type=click.Path(exists=True),
              help="Input data CSV file")
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Fitted model JSON file")
@click.option("-p", "--plot", "plot_path", required=True, type=click.Path(exists=True),
              help="Existing fit diagnostics plot image")
@click.option("-o", "--output", "output_path", type=click.Path(),
              default="./reports/fit_report.md",
              help="Path for the Markdown report (default: ./reports/fit_report.md)")
@click.option("--embed-images", is_flag=True,
              help="Embed plot images as base64 data URIs")
def report_fit(data_path: str, model_path: str, plot_path: str, output_path: str,
               embed_images: bool):
    """Generate a Markdown fit report from existing files."""
    try:
        data = load_csv_data(Path(data_path))
        model = load_model_json(Path(model_path))
        report_file = generate_fit_report(
            data=data,
            model_json=model,
            plot_path=Path(plot_path),
            output_path=Path(output_path),
            use_data_uri=embed_images,
        )
        click.echo(f"✓ Saved Markdown report to: {report_file}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@report.command("forecast")
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Fitted model JSON file")
@click.option("-f", "--forecast", "forecast_path", required=True, type=click.Path(exists=True),
              help="Forecast CSV file")
@click.option("-p", "--plot", "plot_path", required=True, type=click.Path(exists=True),
              help="Existing forecast plot image")
@click.option("-o", "--output", "output_path", type=click.Path(),
              default="./reports/forecast_report.md",
              help="Path for the Markdown report (default: ./reports/forecast_report.md)")
@click.option("--embed-images", is_flag=True,
              help="Embed plot images as base64 data URIs")
def report_forecast(model_path: str, forecast_path: str, plot_path: str,
                    output_path: str, embed_images: bool):
    """Generate a Markdown forecast report from existing files."""
    try:
        model = load_model_json(Path(model_path))
        forecast_df = load_forecast_csv(Path(forecast_path))
        report_file = generate_forecast_report(
            model_json=model,
            forecast_df=forecast_df,
            plot_path=Path(plot_path),
            output_path=Path(output_path),
            use_data_uri=embed_images,
        )
        click.echo(f"✓ Saved Markdown report to: {report_file}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@report.command("diagnostics")
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Fitted model JSON file")
@click.option("-d", "--data", "data_path", required=True, type=click.Path(exists=True),
              help="Input data CSV file")
@click.option("-j", "--diagnostics-json", "diag_json_path", type=click.Path(exists=True),
              help="Diagnostics JSON file (optional)")
@click.option("-p", "--plot", "plot_path", required=True, type=click.Path(exists=True),
              help="Existing residual diagnostics plot image")
@click.option("-o", "--output", "output_path", type=click.Path(),
              default="./reports/diagnostics_report.md",
              help="Path for the Markdown report (default: ./reports/diagnostics_report.md)")
@click.option("--embed-images", is_flag=True,
              help="Embed plot images as base64 data URIs")
def report_diagnostics(model_path: str, data_path: str, diag_json_path: Optional[str],
                        plot_path: str, output_path: str, embed_images: bool):
    """Generate a Markdown diagnostics report from existing files."""
    try:
        model = load_model_json(Path(model_path))
        data = load_csv_data(Path(data_path))
        diag_data = load_diagnostics_json(Path(diag_json_path)) if diag_json_path else None
        report_file = generate_diagnostics_report(
            model_json=model,
            data=data,
            diagnostics_json=diag_data,
            plot_path=Path(plot_path),
            output_path=Path(output_path),
            use_data_uri=embed_images,
        )
        click.echo(f"✓ Saved Markdown report to: {report_file}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@report.command("simulate")
@click.option("-m", "--model", "model_path", required=True, type=click.Path(exists=True),
              help="Fitted model JSON file")
@click.option("-s", "--simulation", "sim_path", required=True, type=click.Path(exists=True),
              help="Simulation CSV file")
@click.option("-p", "--plot", "plot_path", required=True, type=click.Path(exists=True),
              help="Existing simulation plot image")
@click.option("-o", "--output", "output_path", type=click.Path(),
              default="./reports/simulation_report.md",
              help="Path for the Markdown report (default: ./reports/simulation_report.md)")
@click.option("--embed-images", is_flag=True,
              help="Embed plot images as base64 data URIs")
def report_simulate(model_path: str, sim_path: str, plot_path: str,
                    output_path: str, embed_images: bool):
    """Generate a Markdown simulation report from existing files."""
    try:
        model = load_model_json(Path(model_path))
        simulation_data, n_paths, length = parse_simulation_csv(Path(sim_path))
        report_file = generate_simulation_report(
            model_json=model,
            simulation_df=simulation_data,
            plot_path=Path(plot_path),
            output_path=Path(output_path),
            n_paths=n_paths,
            length=length,
            use_data_uri=embed_images,
        )
        click.echo(f"✓ Saved Markdown report to: {report_file}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
