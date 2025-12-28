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
    # Check if ag executable is available
    ag_path = find_ag_executable()
    if ag_path is None:
        click.echo(
            "Warning: ag executable not found. Please build the project or set "
            "the AG_EXECUTABLE environment variable.",
            err=True
        )


@cli.command()
@click.option('-d', '--data', 'data_path', required=True, type=click.Path(exists=True),
              help='Input data file in CSV format')
@click.option('-a', '--arima', required=True,
              help='ARIMA order as p,d,q (e.g., 1,0,1)')
@click.option('-g', '--garch', required=True,
              help='GARCH order as p,q (e.g., 1,1)')
@click.option('-o', '--output', 'output_path', type=click.Path(),
              help='Output model file in JSON format')
@click.option('--plot-dir', type=click.Path(), default='./output',
              help='Directory to save diagnostic plots')
@click.option('--markdown', is_flag=True,
              help='Generate a professional Markdown report with analysis and visuals')
def fit(data_path: str, arima: str, garch: str, output_path: Optional[str], plot_dir: str, markdown: bool):
    """
    Fit ARIMA-GARCH model and generate diagnostic plots.
    
    Examples:
        ag-viz fit -d data.csv -a 1,0,1 -g 1,1 -o model.json
        ag-viz fit -d data.csv -a 1,0,1 -g 1,1 -o model.json --markdown
    """
    click.echo(f"Fitting model: ARIMA({arima})-GARCH({garch})...")
    
    # Default output path if not provided
    if output_path is None:
        output_path = 'model.json'
    
    # Run ag fit command
    try:
        args = [
            'fit',
            '-d', data_path,
            '-a', arima,
            '-g', garch,
            '-o', output_path
        ]
        result = run_ag_command(args)
        click.echo(result.stdout)
        
        # Generate diagnostic plots
        click.echo(f"\nGenerating diagnostic plots in {plot_dir}...")
        data = load_csv_data(Path(data_path))
        model = load_model_json(Path(output_path))
        
        plot_path = plot_fit_diagnostics(data, model, Path(plot_dir))
        click.echo(f"✓ Saved fit diagnostics to: {plot_path}")
        
        # Generate Markdown report if requested
        if markdown:
            report_dir = Path('./reports')
            report_path = report_dir / 'fit_report.md'
            click.echo(f"\nGenerating Markdown report...")
            
            report_file = generate_fit_report(
                data=data,
                model_json=model,
                plot_path=plot_path,
                output_path=report_path,
                use_data_uri=False
            )
            click.echo(f"✓ Saved Markdown report to: {report_file}")
        
        click.echo(f"\n✓ Model saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('-m', '--model', 'model_path', required=True, type=click.Path(exists=True),
              help='Input model file in JSON format')
@click.option('-n', '--horizon', default=10, type=int,
              help='Forecast horizon (number of steps ahead)')
@click.option('-o', '--output', 'output_path', type=click.Path(),
              help='Output forecast file in CSV format')
@click.option('--plot', 'plot_path', type=click.Path(),
              help='Path to save forecast plot')
@click.option('--show', is_flag=True, help='Display the plot')
@click.option('--markdown', is_flag=True,
              help='Generate a professional Markdown report with analysis and visuals')
def forecast(model_path: str, horizon: int, output_path: Optional[str],
            plot_path: Optional[str], show: bool, markdown: bool):
    """
    Generate forecasts and plot with confidence intervals.
    
    Examples:
        ag-viz forecast -m model.json -n 30 -o forecast.csv
        ag-viz forecast -m model.json -n 30 -o forecast.csv --markdown
    """
    click.echo(f"Generating {horizon}-step forecast...")
    
    # Default output path if not provided
    if output_path is None:
        output_path = 'forecast.csv'
    
    # Run ag forecast command
    try:
        args = [
            'forecast',
            '-m', model_path,
            '-n', str(horizon),
            '-o', output_path
        ]
        result = run_ag_command(args)
        click.echo(result.stdout)
        
        # Generate forecast plot
        click.echo("\nGenerating forecast plot...")
        save_path = plot_forecast(
            Path(model_path),
            Path(output_path),
            confidence_levels=[0.68, 0.95],
            show=show,
            save=Path(plot_path) if plot_path else None
        )
        
        if save_path:
            click.echo(f"✓ Saved forecast plot to: {save_path}")
        
        # Generate Markdown report if requested
        if markdown:
            report_dir = Path('./reports')
            report_path = report_dir / 'forecast_report.md'
            click.echo(f"\nGenerating Markdown report...")
            
            model = load_model_json(Path(model_path))
            forecast_data = load_forecast_csv(Path(output_path))
            
            report_file = generate_forecast_report(
                model_json=model,
                forecast_df=forecast_data,
                plot_path=save_path,
                output_path=report_path,
                use_data_uri=False
            )
            click.echo(f"✓ Saved Markdown report to: {report_file}")
        
        click.echo(f"✓ Forecast saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('-m', '--model', 'model_path', required=True, type=click.Path(exists=True),
              help='Input model file in JSON format')
@click.option('-d', '--data', 'data_path', required=True, type=click.Path(exists=True),
              help='Input data file in CSV format')
@click.option('-o', '--output', 'output_dir', type=click.Path(), default='./diagnostics',
              help='Output directory for diagnostic plots and JSON')
@click.option('--markdown', is_flag=True,
              help='Generate a professional Markdown report with analysis and visuals')
def diagnostics(model_path: str, data_path: str, output_dir: str, markdown: bool):
    """
    Generate comprehensive residual diagnostic plots.
    
    Examples:
        ag-viz diagnostics -m model.json -d data.csv -o ./diagnostics/
        ag-viz diagnostics -m model.json -d data.csv -o ./diagnostics/ --markdown
    """
    click.echo("Running diagnostics...")
    
    output_path = Path(output_dir)
    ensure_output_dir(output_path)
    
    # Run ag diagnostics command
    try:
        diag_json = output_path / 'diagnostics.json'
        args = [
            'diagnostics',
            '-m', model_path,
            '-d', data_path,
            '-o', str(diag_json)
        ]
        result = run_ag_command(args)
        click.echo(result.stdout)
        
        # Generate diagnostic plots
        click.echo(f"\nGenerating diagnostic plots in {output_dir}...")
        plot_path = plot_residual_diagnostics(
            Path(model_path),
            Path(data_path),
            diag_json if diag_json.exists() else None,
            output_path
        )
        
        click.echo(f"✓ Saved residual diagnostics to: {plot_path}")
        
        # Generate Markdown report if requested
        if markdown:
            report_dir = Path('./reports')
            report_path = report_dir / 'diagnostics_report.md'
            click.echo(f"\nGenerating Markdown report...")
            
            model = load_model_json(Path(model_path))
            data = load_csv_data(Path(data_path))
            diagnostics_data = load_diagnostics_json(diag_json) if diag_json.exists() else None
            
            report_file = generate_diagnostics_report(
                model_json=model,
                data=data,
                diagnostics_json=diagnostics_data,
                plot_path=plot_path,
                output_path=report_path,
                use_data_uri=False
            )
            click.echo(f"✓ Saved Markdown report to: {report_file}")
        
        click.echo(f"✓ Diagnostics saved to: {diag_json}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('-m', '--model', 'model_path', required=True, type=click.Path(exists=True),
              help='Input model file in JSON format')
@click.option('-p', '--paths', default=100, type=int,
              help='Number of simulation paths to generate')
@click.option('-n', '--length', default=1000, type=int,
              help='Number of observations per path')
@click.option('-s', '--seed', default=42, type=int,
              help='Random seed for reproducibility')
@click.option('-o', '--output', 'output_path', type=click.Path(),
              help='Output simulation file in CSV format')
@click.option('--plot', 'plot_path', type=click.Path(),
              help='Path to save simulation plot')
@click.option('--n-plot', default=10, type=int,
              help='Number of paths to plot')
@click.option('--show', is_flag=True, help='Display the plot')
@click.option('--stats', is_flag=True, help='Compute and display summary statistics')
@click.option('--markdown', is_flag=True,
              help='Generate a professional Markdown report with analysis and visuals')
def simulate(model_path: str, paths: int, length: int, seed: int,
            output_path: Optional[str], plot_path: Optional[str],
            n_plot: int, show: bool, stats: bool, markdown: bool):
    """
    Simulate paths and visualize distributions.
    
    Examples:
        ag-viz simulate -m model.json -p 100 -n 1000 -o simulation.csv
        ag-viz simulate -m model.json -p 100 -n 1000 -o simulation.csv --markdown
    """
    click.echo(f"Simulating {paths} paths with {length} observations each...")
    
    # Default output path if not provided
    if output_path is None:
        output_path = 'simulation.csv'
    
    # Run ag simulate command
    try:
        args = [
            'simulate',
            '-m', model_path,
            '-p', str(paths),
            '-n', str(length),
            '-s', str(seed),
            '-o', output_path
        ]
        
        if stats:
            args.append('--stats')
        
        result = run_ag_command(args)
        click.echo(result.stdout)
        
        # Generate simulation plot
        click.echo("\nGenerating simulation plot...")
        save_path = plot_simulation_paths(
            Path(output_path),
            n_paths_to_plot=n_plot,
            output_path=Path(plot_path) if plot_path else None,
            show=show
        )
        
        click.echo(f"✓ Saved simulation plot to: {save_path}")
        
        # Generate Markdown report if requested
        if markdown:
            report_dir = Path('./reports')
            report_path = report_dir / 'simulation_report.md'
            click.echo(f"\nGenerating Markdown report...")
            
            model = load_model_json(Path(model_path))
            # parse_simulation_csv returns a tuple (df, n_paths, n_obs)
            simulation_data, _, _ = parse_simulation_csv(Path(output_path))
            
            report_file = generate_simulation_report(
                model_json=model,
                simulation_df=simulation_data,
                plot_path=save_path,
                output_path=report_path,
                n_paths=paths,
                length=length,
                use_data_uri=False
            )
            click.echo(f"✓ Saved Markdown report to: {report_file}")
        
        click.echo(f"✓ Simulation data saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
