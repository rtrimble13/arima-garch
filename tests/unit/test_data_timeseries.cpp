#include "ag/data/TimeSeries.hpp"

#include <vector>

#include "test_framework.hpp"

using ag::data::SeriesView;
using ag::data::TimeSeries;

// Test TimeSeries default construction
TEST(timeseries_default_construction) {
    TimeSeries ts;
    REQUIRE(ts.size() == 0);
    REQUIRE(ts.empty());
}

// Test TimeSeries construction from vector
TEST(timeseries_vector_construction) {
    std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0};
    TimeSeries ts(data);
    REQUIRE(ts.size() == 5);
    REQUIRE(!ts.empty());
    REQUIRE(ts[0] == 1.0);
    REQUIRE(ts[4] == 5.0);
}

// Test TimeSeries construction from initializer list
TEST(timeseries_initializer_list_construction) {
    TimeSeries ts{1.0, 2.0, 3.0, 4.0, 5.0};
    REQUIRE(ts.size() == 5);
    REQUIRE(!ts.empty());
    REQUIRE(ts[0] == 1.0);
    REQUIRE(ts[4] == 5.0);
}

// Test TimeSeries size method
TEST(timeseries_size) {
    TimeSeries ts1;
    REQUIRE(ts1.size() == 0);

    TimeSeries ts2{1.0, 2.0, 3.0};
    REQUIRE(ts2.size() == 3);

    TimeSeries ts3{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    REQUIRE(ts3.size() == 10);
}

// Test TimeSeries mean method with small arrays
TEST(timeseries_mean_small_arrays) {
    // Empty time series
    TimeSeries ts_empty;
    REQUIRE_APPROX(ts_empty.mean(), 0.0, 1e-10);

    // Single element
    TimeSeries ts_single{5.0};
    REQUIRE_APPROX(ts_single.mean(), 5.0, 1e-10);

    // Two elements
    TimeSeries ts_two{2.0, 4.0};
    REQUIRE_APPROX(ts_two.mean(), 3.0, 1e-10);

    // Three elements
    TimeSeries ts_three{1.0, 2.0, 3.0};
    REQUIRE_APPROX(ts_three.mean(), 2.0, 1e-10);

    // Five elements
    TimeSeries ts_five{1.0, 2.0, 3.0, 4.0, 5.0};
    REQUIRE_APPROX(ts_five.mean(), 3.0, 1e-10);

    // Elements with negative values
    TimeSeries ts_negative{-2.0, 0.0, 2.0};
    REQUIRE_APPROX(ts_negative.mean(), 0.0, 1e-10);

    // Elements with decimals
    TimeSeries ts_decimals{1.5, 2.5, 3.5, 4.5};
    REQUIRE_APPROX(ts_decimals.mean(), 3.0, 1e-10);
}

// Test TimeSeries element access
TEST(timeseries_element_access) {
    TimeSeries ts{10.0, 20.0, 30.0, 40.0, 50.0};
    REQUIRE(ts[0] == 10.0);
    REQUIRE(ts[1] == 20.0);
    REQUIRE(ts[2] == 30.0);
    REQUIRE(ts[3] == 40.0);
    REQUIRE(ts[4] == 50.0);

    // Test non-const access
    ts[2] = 35.0;
    REQUIRE(ts[2] == 35.0);
}

// Test TimeSeries view creation
TEST(timeseries_view_creation) {
    TimeSeries ts{1.0, 2.0, 3.0, 4.0, 5.0};
    SeriesView view = ts.view();

    REQUIRE(view.size() == 5);
    REQUIRE(!view.empty());
    REQUIRE(view[0] == 1.0);
    REQUIRE(view[4] == 5.0);
}

// Test TimeSeries subview creation
TEST(timeseries_subview_creation) {
    TimeSeries ts{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    SeriesView subview = ts.view(2, 5);

    REQUIRE(subview.size() == 5);
    REQUIRE(subview[0] == 3.0);  // Starting at index 2
    REQUIRE(subview[4] == 7.0);  // 5 elements: indices 2,3,4,5,6
}

// Test SeriesView construction from pointer and size
TEST(seriesview_pointer_construction) {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    SeriesView view(data, 5);

    REQUIRE(view.size() == 5);
    REQUIRE(!view.empty());
    REQUIRE(view[0] == 1.0);
    REQUIRE(view[4] == 5.0);
}

// Test SeriesView construction from vector
TEST(seriesview_vector_construction) {
    std::vector<double> data{10.0, 20.0, 30.0};
    SeriesView view(data);

    REQUIRE(view.size() == 3);
    REQUIRE(view[0] == 10.0);
    REQUIRE(view[2] == 30.0);
}

// Test SeriesView size and empty
TEST(seriesview_size_empty) {
    double data[] = {1.0, 2.0, 3.0};
    SeriesView view1(data, 3);
    REQUIRE(view1.size() == 3);
    REQUIRE(!view1.empty());

    SeriesView view2(data, 0);
    REQUIRE(view2.size() == 0);
    REQUIRE(view2.empty());
}

// Test SeriesView mean method
TEST(seriesview_mean) {
    // Empty view
    double empty_data[] = {0.0};
    SeriesView view_empty(empty_data, 0);
    REQUIRE_APPROX(view_empty.mean(), 0.0, 1e-10);

    // Single element
    double single_data[] = {7.0};
    SeriesView view_single(single_data, 1);
    REQUIRE_APPROX(view_single.mean(), 7.0, 1e-10);

    // Multiple elements
    double data[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    SeriesView view(data, 5);
    REQUIRE_APPROX(view.mean(), 6.0, 1e-10);

    // Negative values
    double neg_data[] = {-5.0, 0.0, 5.0};
    SeriesView view_neg(neg_data, 3);
    REQUIRE_APPROX(view_neg.mean(), 0.0, 1e-10);
}

// Test SeriesView element access
TEST(seriesview_element_access) {
    double data[] = {100.0, 200.0, 300.0, 400.0};
    SeriesView view(data, 4);

    REQUIRE(view[0] == 100.0);
    REQUIRE(view[1] == 200.0);
    REQUIRE(view[2] == 300.0);
    REQUIRE(view[3] == 400.0);
}

// Test SeriesView correctness with TimeSeries
TEST(seriesview_correctness_with_timeseries) {
    TimeSeries ts{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    // Full view
    SeriesView full_view = ts.view();
    REQUIRE(full_view.size() == ts.size());
    REQUIRE_APPROX(full_view.mean(), ts.mean(), 1e-10);

    // Subview
    SeriesView sub_view = ts.view(3, 4);  // Elements 4, 5, 6, 7
    REQUIRE(sub_view.size() == 4);
    REQUIRE(sub_view[0] == 4.0);
    REQUIRE(sub_view[3] == 7.0);
    REQUIRE_APPROX(sub_view.mean(), 5.5, 1e-10);  // (4+5+6+7)/4 = 5.5
}

// Test SeriesView iteration
TEST(seriesview_iteration) {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    SeriesView view(data, 5);

    double sum = 0.0;
    for (double value : view) {
        sum += value;
    }
    REQUIRE_APPROX(sum, 15.0, 1e-10);
}

// Test TimeSeries iteration
TEST(timeseries_iteration) {
    TimeSeries ts{2.0, 4.0, 6.0, 8.0, 10.0};

    double sum = 0.0;
    for (double value : ts) {
        sum += value;
    }
    REQUIRE_APPROX(sum, 30.0, 1e-10);
}

// Test that view reflects underlying data
TEST(seriesview_reflects_underlying_data) {
    std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0};
    SeriesView view(data);

    // Initial mean
    REQUIRE_APPROX(view.mean(), 3.0, 1e-10);

    // Modify underlying data
    data[2] = 10.0;  // Change 3.0 to 10.0

    // View should reflect the change
    REQUIRE(view[2] == 10.0);
    REQUIRE_APPROX(view.mean(), 4.4, 1e-10);  // (1+2+10+4+5)/5 = 4.4
}

int main() {
    report_test_results("TimeSeries and SeriesView Tests");
    return get_test_result();
}
