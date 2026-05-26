/**
 * @file cli/main.cpp
 * @brief Command-line interface for ARIMA-GARCH modeling.
 *
 * Argument parsing and subcommand dispatch. The actual command
 * implementations live under src/cli/handlers/.
 */

#include "ag/Version.hpp"

#include <string>

#include "Handlers.hpp"
#include <CLI/CLI.hpp>

int main(int argc, char* argv[]) {
    CLI::App app{"ARIMA-GARCH Time Series Modeling CLI", "ag"};
    app.require_subcommand(0, 1);
    app.set_version_flag("--version,-v", ag::kVersion);

    // ----- fit -----
    auto* fit = app.add_subcommand("fit", "Fit ARIMA-GARCH model to time series data");
    std::string fit_data_file;
    std::string fit_arima_order;
    std::string fit_garch_order;
    std::string fit_output_file;
    bool fit_no_header = false;
    double fit_student_t_df = 0.0;

    fit->add_option("-d,--data", fit_data_file,
                    "Input data file (CSV format, auto-detects first numeric column)")
        ->required();
    fit->add_option("-a,--arima", fit_arima_order, "ARIMA order as p,d,q (e.g., 1,1,1)");
    fit->add_option("-g,--garch", fit_garch_order, "GARCH order as p,q (e.g., 1,1)");
    fit->add_option("-o,--output,--out", fit_output_file, "Output model file (JSON format)");
    fit->add_flag("--no-header", fit_no_header,
                  "CSV file has no header row (default: expect header)");
    fit->add_option("--t-dist", fit_student_t_df,
                    "Use Student-t distribution with specified degrees of freedom (default: 2.0)")
        ->check(CLI::PositiveNumber);
    fit->callback([&]() {
        const bool use_student_t = fit->count("--t-dist") > 0;
        const double df = use_student_t ? fit_student_t_df : 2.0;
        return ag::cli::handleFit(fit_data_file, fit_arima_order, fit_garch_order, fit_output_file,
                                  fit_no_header, use_student_t, df);
    });

    // ----- select -----
    auto* select = app.add_subcommand("select", "Automatic model selection from candidate grid");
    std::string select_data_file;
    int select_max_p = 2;
    int select_max_d = 1;
    int select_max_q = 2;
    int select_max_garch_p = 1;
    int select_max_garch_q = 1;
    std::string select_criterion = "BIC";
    std::string select_output_file;
    int select_top_k = 0;
    bool select_no_header = false;

    select
        ->add_option("-d,--data", select_data_file,
                     "Input data file (CSV format, auto-detects first numeric column)")
        ->required();
    select->add_option("--max-p", select_max_p, "Maximum ARIMA AR order (default: 2)");
    select->add_option("--max-d", select_max_d, "Maximum ARIMA differencing order (default: 1)");
    select->add_option("--max-q", select_max_q, "Maximum ARIMA MA order (default: 2)");
    select->add_option("--max-garch-p", select_max_garch_p, "Maximum GARCH p order (default: 1)");
    select->add_option("--max-garch-q", select_max_garch_q, "Maximum GARCH q order (default: 1)");
    select->add_option("-c,--criterion", select_criterion,
                       "Selection criterion: BIC, AIC, AICc, or CV (default: BIC)");
    select->add_option("-o,--output,--out", select_output_file, "Output model file (JSON format)");
    select->add_option("--top-k", select_top_k,
                       "Display top K models in ranking table (default: 0, disabled)");
    select->add_flag("--no-header", select_no_header,
                     "CSV file has no header row (default: expect header)");
    select->callback([&]() {
        return ag::cli::handleSelect(select_data_file, select_max_p, select_max_d, select_max_q,
                                     select_max_garch_p, select_max_garch_q, select_criterion,
                                     select_output_file, select_top_k, select_no_header);
    });

    // ----- forecast -----
    auto* forecast = app.add_subcommand("forecast", "Generate forecasts from fitted model");
    std::string forecast_model_file;
    int forecast_horizon = 10;
    std::string forecast_output_file;

    forecast->add_option("-m,--model", forecast_model_file, "Input model file (JSON format)")
        ->required();
    forecast->add_option("-n,--horizon", forecast_horizon,
                         "Forecast horizon (number of steps ahead, default: 10)");
    forecast->add_option("-o,--output,--out", forecast_output_file,
                         "Output forecast file (CSV format)");
    forecast->callback([&]() {
        return ag::cli::handleForecast(forecast_model_file, forecast_horizon, forecast_output_file);
    });

    // ----- sim (synthetic from spec) -----
    auto* simulate = app.add_subcommand("sim", "Simulate synthetic time series data");
    std::string sim_arima_order;
    std::string sim_garch_order;
    int sim_length = 1000;
    unsigned int sim_seed = 42;
    std::string sim_output_file;
    double sim_student_t_df = 0.0;

    simulate->add_option("-a,--arima", sim_arima_order, "ARIMA order as p,d,q (e.g., 1,1,1)")
        ->required();
    simulate->add_option("-g,--garch", sim_garch_order, "GARCH order as p,q (e.g., 1,1)")
        ->required();
    simulate->add_option("-n,--length", sim_length,
                         "Number of observations to simulate (default: 1000)");
    simulate->add_option("-s,--seed", sim_seed, "Random seed (default: 42)");
    simulate->add_option("-o,--output,--out", sim_output_file, "Output data file (CSV format)")
        ->required();
    simulate
        ->add_option("--t-dist", sim_student_t_df,
                     "Use Student-t distribution with specified degrees of freedom (default: 2.0)")
        ->check(CLI::PositiveNumber);
    simulate->callback([&]() {
        const bool use_student_t = simulate->count("--t-dist") > 0;
        const double df = use_student_t ? sim_student_t_df : 2.0;
        return ag::cli::handleSimulate(sim_arima_order, sim_garch_order, sim_length, sim_seed,
                                       sim_output_file, use_student_t, df);
    });

    // ----- simulate (multi-path from saved model) -----
    auto* simulate_model =
        app.add_subcommand("simulate", "Simulate multiple paths from a saved model");
    std::string simmodel_model_file;
    int simmodel_num_paths = 1;
    int simmodel_length = 1000;
    unsigned int simmodel_seed = 42;
    std::string simmodel_output_file;
    bool simmodel_compute_stats = false;

    simulate_model->add_option("-m,--model", simmodel_model_file, "Input model file (JSON format)")
        ->required();
    simulate_model->add_option("-p,--paths", simmodel_num_paths,
                               "Number of simulation paths (default: 1)");
    simulate_model->add_option("-n,--length", simmodel_length,
                               "Number of observations per path (default: 1000)");
    simulate_model->add_option("-s,--seed", simmodel_seed, "Random seed (default: 42)");
    simulate_model
        ->add_option("-o,--output,--out", simmodel_output_file,
                     "Output CSV file (e.g., sim_returns.csv)")
        ->required();
    simulate_model->add_flag("--stats", simmodel_compute_stats,
                             "Compute and display summary statistics");
    simulate_model->callback([&]() {
        return ag::cli::handleSimulateFromModel(simmodel_model_file, simmodel_num_paths,
                                                simmodel_length, simmodel_seed,
                                                simmodel_output_file, simmodel_compute_stats);
    });

    // ----- diagnostics -----
    auto* diagnostics = app.add_subcommand("diagnostics", "Run diagnostic tests on fitted model");
    std::string diag_model_file;
    std::string diag_data_file;
    std::string diag_output_file;
    bool diag_no_header = false;

    diagnostics->add_option("-m,--model", diag_model_file, "Input model file (JSON format)")
        ->required();
    diagnostics
        ->add_option("-d,--data", diag_data_file,
                     "Input data file (CSV format, auto-detects first numeric column)")
        ->required();
    diagnostics->add_option("-o,--output,--out", diag_output_file,
                            "Output diagnostics file (JSON format)");
    diagnostics->add_flag("--no-header", diag_no_header,
                          "CSV file has no header row (default: expect header)");
    diagnostics->callback([&]() {
        return ag::cli::handleDiagnostics(diag_model_file, diag_data_file, diag_output_file,
                                          diag_no_header);
    });

    CLI11_PARSE(app, argc, argv);
    return 0;
}
