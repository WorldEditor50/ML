#ifndef BAYES_H
#define BAYES_H
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
namespace ML {
    class Bayes {
        public:
            Bayes(){}
            ~Bayes(){}
            void load(const std::string& fileName);
            std::string classify(std::string& sample);
            void show();
            void calculatePrior();
        private:
            void splitString(std::string& strSrc, std::vector<std::string>& strVect, const std::string& pattern);
            void convertDataToFeature(const std::vector<std::string>& xi, std::vector<int>& feature);
            int indicator(int x_id, int y_id);
            void calculatePosterior(std::vector<int>& xi);
            int elementNum;
            std::vector<std::string> elements;
            std::vector<std::string> labels;
            std::vector<std::vector<int> > x;
            std::vector<int> y;
            std::vector<double> margin;
            std::vector<std::vector<double> > condition;
            std::vector<double> posterior;
    };
}
#endif// BAYES_H
