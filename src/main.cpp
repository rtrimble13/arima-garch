#include "ag/arima_garch.hpp"
#include "ag/report/FitSummary.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "ARIMA-GARCH CLI v0.1.0" << std::endl;
    std::cout << "Usage: ag [command] [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  fit        Fit ARIMA-GARCH model to data (outputs FitSummary)" << std::endl;
    std::cout << "  forecast   Forecast future values" << std::endl;
    std::cout << "  simulate   Simulate synthetic data" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: The 'fit' command will generate a comprehensive FitSummary" << std::endl;
    std::cout << "      report including model specification, parameters, convergence" << std::endl;
    std::cout << "      info, AIC/BIC/loglik, and diagnostic tests." << std::endl;
    std::cout << std::endl;
    std::cout << "See examples/example_fit_summary.cpp for a complete demonstration." << std::endl;
    return 0;
}
