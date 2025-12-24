#pragma once

#include "ag/api/Engine.hpp"
#include "ag/data/TimeSeries.hpp"
#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/diagnostics/Residuals.hpp"
#include "ag/estimation/Constraints.hpp"
#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/NumericalDerivatives.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/forecasting/Forecaster.hpp"
#include "ag/io/CsvReader.hpp"
#include "ag/io/CsvWriter.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/ArimaSpec.hpp"
#include "ag/models/GarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/arima/ArimaState.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/models/garch/GarchModel.hpp"
#include "ag/models/garch/GarchState.hpp"
#include "ag/stats/ACF.hpp"
#include "ag/stats/ADF.hpp"
#include "ag/stats/Descriptive.hpp"
#include "ag/stats/JarqueBera.hpp"
#include "ag/stats/LjungBox.hpp"
#include "ag/stats/PACF.hpp"
#include "ag/util/Expected.hpp"
#include "ag/util/Logging.hpp"
#include "ag/util/Timer.hpp"

namespace ag {

// ARIMA-GARCH library for time series modeling
// Provides functionality for fitting, forecasting, and analyzing
// ARIMA-GARCH models

}  // namespace ag
