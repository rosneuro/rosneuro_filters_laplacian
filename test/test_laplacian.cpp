#include <gtest/gtest.h>
#include "Laplacian.hpp"

namespace rosneuro {

class LaplacianTestSuite : public ::testing::Test {
    public:
        LaplacianTestSuite() { laplacian_filter = new Laplacian <double>(); }
        ~LaplacianTestSuite() { delete laplacian_filter; }
        Laplacian <double>* laplacian_filter = new Laplacian <double>();
};

TEST_F(LaplacianTestSuite, Constructor) {
    ASSERT_EQ(laplacian_filter->name_, "laplacian");
    ASSERT_TRUE(laplacian_filter->is_mask_set_);
}

TEST_F(LaplacianTestSuite, Configure) {
    ASSERT_FALSE(laplacian_filter->configure());

    std::string matrixString = " 0   0   1   0   2   0   0;\n"
                               " 0   0   0   0   0   0   0;\n"
                               " 0   0  18   3  19   0   0;\n"
                               " 4  20   5  21   6  22   7;\n"
                               "23   8  24   9  25  10  26;\n"
                               "11  27  12   0  13  28  14;\n"
                               "29  15  30  16  31  17  32";

    laplacian_filter->params_["layout"] = XmlRpc::XmlRpcValue(matrixString);
    ASSERT_TRUE(laplacian_filter->configure());
    ASSERT_TRUE(laplacian_filter->is_mask_set_);

    laplacian_filter->params_["nchannels"] = XmlRpc::XmlRpcValue(8);
    ASSERT_FALSE(laplacian_filter->configure());
    ASSERT_TRUE(laplacian_filter->is_mask_set_);

    std::string duplicatedIndices = "1 1";
    laplacian_filter->params_["layout"] = XmlRpc::XmlRpcValue(duplicatedIndices);
    laplacian_filter->params_.erase("nchannels");
    ASSERT_FALSE(laplacian_filter->configure());

    std::string invalidLayout = "1 2 3; 4 5; 7 8 9";
    laplacian_filter->params_["layout"] = XmlRpc::XmlRpcValue(invalidLayout);
    ASSERT_FALSE(laplacian_filter->configure());
}

TEST_F(LaplacianTestSuite, SetLayoutDynamicMatrix) {
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> layout(3, 3);
    int nchannels = 3;

    ASSERT_TRUE(laplacian_filter->set_layout(layout, nchannels));
    ASSERT_TRUE(laplacian_filter->is_mask_set_);
}

TEST_F(LaplacianTestSuite, SetLayoutString) {
    std::string validLayout = "1 2 3; 4 5 6; 7 8 9";
    int nchannels = 3;

    ASSERT_TRUE(laplacian_filter->set_layout(validLayout, nchannels));
    ASSERT_TRUE(laplacian_filter->is_mask_set_);
}

TEST_F(LaplacianTestSuite, SetLayoutStringWithDuplicates) {
    std::string invalidLayout = "1 2 3; 4 5; 7 8 9";
    int nchannels = 3;

    ASSERT_FALSE(laplacian_filter->set_layout(invalidLayout, nchannels));
    ASSERT_FALSE(laplacian_filter->is_mask_set_);
}

TEST_F(LaplacianTestSuite, SetMask) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mask(3, 3);

    ASSERT_TRUE(laplacian_filter->set_mask(mask));
    ASSERT_TRUE(laplacian_filter->is_mask_set_);
    ASSERT_EQ(laplacian_filter->mask(), mask);
}

TEST_F(LaplacianTestSuite, CreateMask) {
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> layout(3, 3);
    int nchannels = 3;

    ASSERT_TRUE(laplacian_filter->set_layout(layout, nchannels));
    ASSERT_TRUE(laplacian_filter->create_mask());
    ASSERT_TRUE(laplacian_filter->is_mask_set_);
}

TEST_F(LaplacianTestSuite, FindChannel) {
    laplacian_filter->set_layout("1 2 3; 4 5 6; 7 8 9", 3);

    unsigned int rowId, colId;

    ASSERT_TRUE(laplacian_filter->find_channel(5, rowId, colId));
    ASSERT_EQ(rowId, 1);
    ASSERT_EQ(colId, 1);

    ASSERT_FALSE(laplacian_filter->find_channel(10, rowId, colId));
}

TEST_F(LaplacianTestSuite, GetNeighboursAllSides) {
    laplacian_filter->set_layout("1 2 3; 4 5 6; 7 8 9", 3);
    std::vector<int> neighbors = laplacian_filter->get_neightbours(1, 1);

    ASSERT_EQ(neighbors.size(), 4);
    ASSERT_EQ(neighbors[0], 4);
    ASSERT_EQ(neighbors[1], 6);
    ASSERT_EQ(neighbors[2], 2);
    ASSERT_EQ(neighbors[3], 8);
}

TEST_F(LaplacianTestSuite, GetNeighboursNoNeighbors) {
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> layout(3, 3);
    layout << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    laplacian_filter->set_layout(layout, 3);
    std::vector<int> neighbors = laplacian_filter->get_neightbours(0, 0);
    ASSERT_EQ(neighbors.size(), 2);
}

TEST_F(LaplacianTestSuite, GetNeighboursTopEdge) {
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> layout(3, 3);
    layout << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    laplacian_filter->set_layout(layout, 3);
    std::vector<int> neighbors = laplacian_filter->get_neightbours(0, 1);

    ASSERT_EQ(neighbors.size(), 3);
    ASSERT_EQ(neighbors[0], 1);
    ASSERT_EQ(neighbors[1], 3);
    ASSERT_EQ(neighbors[2], 5);
}

TEST_F(LaplacianTestSuite, ApplyWithMaskSet) {
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> layout(3, 3);
    layout << 1, 15, 30, 16, 31, 17, 32, 33, 34;
    int nchannels = 3;
    laplacian_filter->set_layout(layout, nchannels);
    laplacian_filter->create_mask();

    DynamicMatrix<double> in(3, 3);
    in << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    DynamicMatrix<double> expected_result(3, 3);
    expected_result << 1, 0, 0, 4, 0, 0, 7, 0, 0;

    EXPECT_EQ(laplacian_filter->apply(in), expected_result);
}

TEST_F(LaplacianTestSuite, ApplyWithoutMaskSet) {
    DynamicMatrix<double> in(3, 3);
    in << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    laplacian_filter->is_mask_set_ = false;
    ASSERT_THROW(laplacian_filter->apply(in), std::runtime_error);
}

TEST_F(LaplacianTestSuite, LoadLayoutValid) {
    std::string valid_layout = "1 2 3; 4 5 6; 7 8 9";
    ASSERT_TRUE(laplacian_filter->load_layout(valid_layout));
    ASSERT_EQ(laplacian_filter->layout_.rows(), 3);
    ASSERT_EQ(laplacian_filter->layout_.cols(), 3);
}

TEST_F(LaplacianTestSuite, LoadLayoutInvalid) {
    std::string invalid_layout = "1 2 3; 4 5; 7 8 9";
    ASSERT_FALSE(laplacian_filter->load_layout(invalid_layout));
}

}

int main(int argc, char **argv) {
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal);
    ros::init(argc, argv, "test_laplacian");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}