#include "LogisticRegression.h"

#include <fstream>
#include <iostream>

/**
 * Example of using the LogisticRegression class for a loan approval.
 * Requires two data files. A training file of samples and
 * a label file for the corresponding sample in the training file.
 * Training file has credit score, annual income, and age.
 * Lable file has one label for each corresponding row of the data file.
 * 
 * @author Bruce Blum
 */

using namespace std;

int main(){
    string trainDataFName;
    string labelDataFName;
    unsigned int miniBatchSize{100};    //Number of samples to use per weight update.
    unsigned int numOfIterations{100};   //Number of updates to weights.
    
    vector<vector<double>> trainingSetVec;  //Holds training data.
    vector<int> labelVec;   //Holds label data.

    vector<double> testVect;
    double loanAmt{0.0};
    double creditScore{0};
    double annualIncome{0};
    double age{0};
    int loanGranted{0}; // 0 for not given loan and 1 for granted a loan
    double enteredLearningRate{0.0001};
    unsigned int maxIters{1000};

    cout << "Example of using the LogisticRegression class." << endl;
    cout << "Enter training data file name or T(rainingData.txt): ";
    cin >> trainDataFName;
    if(trainDataFName.compare("T") == 0){
        trainDataFName = "TrainingData.txt";
    }

    ifstream insTrain{trainDataFName};
    if(!insTrain){ 
        cout << "Not able to open data file " << trainDataFName;
        return 0;
    }

    cout << "Enter target data file name or L(LabelsData.txt): ";
    cin >> labelDataFName;
    if(labelDataFName.compare("L") == 0){
        labelDataFName = "LabelsData.txt";
    }

    ifstream insLabel{labelDataFName};
    if(!insLabel){
        cout << "Not able to open data file " << labelDataFName;
        return 0;
    }

    cout << endl;

    // Read in the training data file and put each row into a vector.
    trainingSetVec.clear();

    while(!insTrain.eof()) {
        vector<double> dataRow;
        
        if(insTrain >> creditScore >> annualIncome >> age) {
        //if(insTrain >> loanAmt){
            //dataRow.push_back(loanAmt);
            dataRow.push_back(creditScore);
            dataRow.push_back(annualIncome);
            dataRow.push_back(age);
            //cout << LogisticRegression::dblVectToString(dataRow) << endl;
            trainingSetVec.push_back(dataRow);
        } else {
            break;
        }
    }

    cout << "Training vector size: " << trainingSetVec.size() << endl;

    // Read in the label data file and put it in a vector.
    labelVec.clear();

    while(!insLabel.eof()){
        if(insLabel >> loanGranted) {
            labelVec.push_back(loanGranted);
        } else{
            break;
        }
    }

    cout << "Label vector size: " << labelVec.size() << endl;

    if(trainingSetVec.size() != labelVec.size()){
        cout << "Training and corresponding label data is not the same size." << endl;
        exit(1);
    }

    LogisticRegression* logReg = new LogisticRegression();

    cout << "Enter learning rate or 0 for default of " << logReg->getLearningRate() << ": ";
    cin >> enteredLearningRate;
    
    if(enteredLearningRate != 0.0){
        logReg->setLearningRate(enteredLearningRate);
    }
    cout << "Learning rate: " << logReg->getLearningRate() << endl;

    cout << "Enter max iterations: ";
    cin >> maxIters;

    logReg->loadTrainingData(trainingSetVec);
    logReg->loadLabelData(labelVec);
    logReg->initRandomWeights(trainingSetVec.at(0).size());
    logReg->gradientAscentWeightUpdate(maxIters);

    string weightsStr = LogisticRegression::dblVectToString(logReg->getWeights());
    cout << "Trained Weights: " << weightsStr << endl;
    cout << "Log Likelyhood: " << logReg->getCurrentLogLikelyhood() << endl;

    vector<double> classifyVec;
    while(true) {
        cout << "Classify a person for a loan." << endl;
        cout << "Enter Credit Score: ";
        cin >> creditScore;
        classifyVec.push_back(creditScore);

        cout << "Enter Annual Income In Thousands: ";
        cin >> annualIncome;
        classifyVec.push_back(annualIncome);

        cout << "Enter Age: ";
        cin >> age;
        classifyVec.push_back(age);
    
        double classResult = logReg->classify(classifyVec);
        cout << "Classification Result: " << classResult << endl;
        if(classResult >= 0.5){
            cout << "Loan Approved." << endl;
        } else {
            cout << "Loan Denied." << endl;
        }
    }



    return 0;
}
