#define _USE_MATH_DEFINES

#include "LogisticRegression.h"

#include <cfloat>
#include <cmath>
#include <iostream>
#include <random>

/**
 * Implementation of the logistic regression using gradient ascent.
 * 
 * @author Bruce Blum
 **/

LogisticRegression::LogisticRegression(){

}

void LogisticRegression::addBiasElementToTrainingVector(){
    for(unsigned int i = 0; i < trainingVec.size(); i++){
        trainingVec.at(i).push_back(1.0);
    }
}

double LogisticRegression::classify(std::vector<double> dataVec){
    double classificationResult{0.0};

    dataVec.push_back(1.0);

    classificationResult = logisticFunction(weightVec, dataVec);

    return classificationResult;
}

std::string LogisticRegression::dblVectToString(const std::vector<double>& dVec) {
    std::string vectStr;
    for(int i = 0; i < dVec.size(); i++){
        vectStr += std::to_string(dVec.at(i));
        vectStr += " ";
    }

    return vectStr;
}

int LogisticRegression::estimate(unsigned long trainingIndex){
    
}

double LogisticRegression::getCurrentLogLikelyhood(){
    return currentLogLikelyhood;
}

double LogisticRegression::getLearningRate(){
    return learningRate;
}

std::vector<double> LogisticRegression::getWeights(){
    return weightVec;
}

/**
 * Update each weight by performing gradient ascent. Stop when log likelyhood is
 * max.
 **/
void LogisticRegression::gradientAscentWeightUpdate(unsigned int maxIterations){
    std::cout << "gradientAscentWeightUpdate()" << std::endl;

    double previousLogLike = -1 * DBL_MAX;

    for(unsigned int i = 0; i < maxIterations; i++){
        std::cout << "previousLogLike: " << previousLogLike << std::endl;
        for(unsigned int j = 0; j < weightVec.size(); j++){
            std::cout << "weight " << j << ": " << weightVec.at(j) << std::endl;
            weightVec.at(j) = weightVec.at(j) + learningRate * sumDifferenceTimesTraining(j);
            std::cout << "weight " << j << ": " << weightVec.at(j) << std::endl;
        }

        currentLogLikelyhood = logLikelyhoodSum();
        std::cout << "currentLogLikelyhood: " << currentLogLikelyhood << std::endl;
        if((currentLogLikelyhood > (-1*DBL_MAX)) && (previousLogLike > currentLogLikelyhood)){
            std::cout << "previousLogLike > currentLogLikelyhood break out of iteration loop." << std::endl;
            break;
        }

        previousLogLike = currentLogLikelyhood;
    }
}

void LogisticRegression::initRandomWeights(unsigned int size){
    std::cout << "initRandomWeights()" << std::endl;
    weightVec.clear();
    std::random_device randDev;
    std::default_random_engine generator(randDev());
    std::uniform_real_distribution<double> uniDist(0.0, 1.0);

    for(unsigned int i = 0; i < size + 1; i++){
        double randWeight = uniDist(generator);
        weightVec.push_back(randWeight);
        std::cout << randWeight << " ";
    }
    std::cout << std::endl;
}

void LogisticRegression::loadLabelData(std::vector<int> dataVec){
    labelVec = dataVec;
}

void LogisticRegression::loadTrainingData(std::vector<std::vector<double>> dataVec){
    trainingVec = dataVec;

    // Add a "1.0" to the end of each training sample.
    addBiasElementToTrainingVector();
}

void LogisticRegression::loadWeights(std::vector<double> weights){
    weightVec = weights;
}

double LogisticRegression::logisticFunction(std::vector<double> weightVec, std::vector<double> xVec){
    std::cout << "logisticFunction()" << LogisticRegression::dblVectToString(xVec) << std::endl;
    double result{0.0};
    double sumWeightAndX{0.0};

    for(int i = 0; i < weightVec.size(); i++){
        sumWeightAndX += weightVec.at(i) * xVec.at(i);
    }

    sumWeightAndX = static_cast<double>(-1.0) * sumWeightAndX;
    std::cout << "sumWeightAndX: " << sumWeightAndX << std::endl;

    result = static_cast<double>(1.0) / (static_cast<double>(1.0) + pow(M_E, sumWeightAndX));
    std::cout << "result: " << result << std::endl;
    return result;
}

double LogisticRegression::logLikelyhoodSum(){
    double sum{0.0};

    for(unsigned int i = 0; i < trainingVec.size(); i++){
        double logisticFncResult = logisticFunction(weightVec, trainingVec.at(i));
        std::cout << "logisticFncResult: " << logisticFncResult << std::endl;
        std::cout << "lableVect.at(" << i << "): " << labelVec.at(i) << std::endl;
        if(labelVec.at(i) == 1){
            std::cout << "label == 1" << std::endl;
            sum += log(logisticFncResult);
        } else{
            std::cout << "label != 1 " << std::endl;
            sum += log(1-logisticFncResult);
        }
        std::cout << "sum: " << sum << std::endl;
    }    
    
    std::cout << "sum: " << sum << std::endl;
    return sum;
}

void LogisticRegression::setLearningRate(double rate){
    learningRate = rate;
}

double LogisticRegression::sumDifferenceTimesTraining(unsigned int weigthIndex){
    std::cout << "sumDifferenceTimesTraining()" << std::endl;
    double sum{0.0};

    for(unsigned int i = 0; i < trainingVec.size(); i++){
        sum += (labelVec.at(i) - logisticFunction(weightVec, trainingVec.at(i))) 
                    * trainingVec.at(i).at(weigthIndex);
    }

    std::cout << "sum: " << sum << std::endl;
    return sum;
}