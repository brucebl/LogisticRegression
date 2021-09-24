#include <string>
#include <vector>

/**
 * Implementation of the binary logistic regression algorithm.
 * Requires two data files. One training file and one label
 * file with the classification for the corresponding row in the
 * training file. 
 * 
 * For example, a loan application of credit score, annual income,
 * and age.
 * Training file:
 * 720  70  51
 * 490  31  28
 * 
 * Label file:
 * 1
 * 0
 * 
 * @author Bruce Blum
 **/

class LogisticRegression{
    public:
        LogisticRegression();

        /**
         * Helper function so make a vector of doubles into a string.
         * 
         * @param dVec vector to make a string of values from.
         **/
        std::string static dblVectToString(const std::vector<double>& dVec);

        void addBiasElementToTrainingVector();

        /**
         * Uses the trained weights to make a classification on the values 
         * of the passed in vector.
         * 
         * @param dataVect holds the values used in the logistic function to
         *          make a classification.
         **/
        double classify(std::vector<double> dataVec);

        int estimate(unsigned long trainingIndex);

        /**
         * Returns the last calculated log likelyhood.
         **/
        double getCurrentLogLikelyhood();

        double getLearningRate();

        std::vector<double> getWeights();

        /**
         * Performs gradient ascent on the current training data and weights.
         * 
         * @param maxIterations the max number of update steps for the weights.
         **/
        void gradientAscentWeightUpdate(unsigned int maxIterations = 100);

        /**
         * Sets the weights to a random value [0, 1].
         * 
         * @param size of the weight vector.
         **/
        void initRandomWeights(unsigned int size = 2);

        /**
         * Loads the label data for the corresponding training data.
         * 
         * @param labels for the training data.
         **/
        void loadLabelData(std::vector<int> dataVec);

        /**
         * Loads the training data.
         * 
         * @param dataVec of training data. Each containing vector will have a 
         *      "1.0" added to it for the bias weight to pair with.
         **/
        void loadTrainingData(std::vector<std::vector<double>> dataVec);

        /**
         * Provides a way to set the weights to specific values usually from
         * a previous training session.
         * 
         * @param weights vector should be sized already to match the training
         *      vector with a bias element.
         **/
        void loadWeights(std::vector<double> weights);

        /**
         * Performs the logistic function on a given weight and data vector.
         * 
         * @param weightVec should match the size of the data vector.
         * @param xVec should match the size of the weight vector.
         **/
        double logisticFunction(std::vector<double> weightVec, std::vector<double> xVec);

        /**
         * Perform the sum of the log likelyhood using current weights.
         **/
        double logLikelyhoodSum();

        void setLearningRate(double rate);

        double sumDifferenceTimesTraining(unsigned int weigthIndex);

    private:
        double currentLogLikelyhood{0.0};
        std::vector<int> labelVec;
        double learningRate{0.00001};
        std::vector<std::vector<double>> trainingVec;
        std::vector<double> weightVec;
};