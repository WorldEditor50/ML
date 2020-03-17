#include "bayes.h"
namespace ML {
    void Bayes::splitString(std::string& strSrc, std::vector<std::string>& strVect, const std::string& pattern)
    {
        unsigned int pos = 0;
        strSrc += pattern;
        unsigned int size = strSrc.size();
        for (unsigned int i = 0; i < size; i++) {
            pos = strSrc.find(pattern, i);
            if (pos < size) {
                std::string strTerm = strSrc.substr(i, pos - i);
                strVect.push_back(strTerm);
                i = pos + pattern.size() - 1;
            }
        }
        return;
    }

    void Bayes::convertDataToFeature(const std::vector<std::string>& xsi, std::vector<int>& xi)
    {
        if (elements.empty()) {
            return;
        }
        for (int i = 0; i < xsi.size(); i++) {
            for (int j = 0; j < elements.size(); j++) {
                if (xsi[i] == elements[j]) {
                    xi[j] = 1;
                    break;
                }
            }
        }
        return;
    }

    void Bayes::load(const std::string& fileName)
    {
        /* load data from file */
        std::string strRow;
        std::ifstream file;
        file.open(fileName);
        std::vector<std::vector<std::string> > xs; 
        while (std::getline(file, strRow)) {
            std::vector<std::string> strVect;
            splitString(strRow, strVect, " ");
            xs.push_back(strVect);
        }
        /* create feature set */
        std::set<std::string> elementSet;
        std::set<std::string> labelSet;
        this->elementNum = 0;
        for (int i = 0; i < xs.size(); i++) {
            for (int j = 0; j < xs[i].size(); j++) {
                if (j < xs[i].size() - 1) {
                    elementSet.insert(xs[i][j]);
                    elementNum++;
                } else {
                    labelSet.insert(xs[i][j]);
                }
            }
        }
        this->elements.assign(elementSet.begin(), elementSet.end());
        this->labels.assign(labelSet.begin(), labelSet.end());
        /* create feature space */
        this->x.resize(xs.size());
        this->y.resize(xs.size());
        for (int i = 0; i < xs.size(); i++) {
            x[i].resize(elements.size(), 0);
            convertDataToFeature(xs[i], x[i]);
            int k = xs[i].size() - 1;
            for (int j = 0; j < labels.size(); j++) {
                if (xs[i][k] == labels[j]) {
                    y[i] = j;
                    break;
                }
            }
        }
        /* proability */
        this->margin.resize(labels.size());
        this->posterior.resize(labels.size());
        this->condition.resize(labels.size());
        for (int i = 0; i < labels.size(); i++) {
            this->condition[i].resize(elements.size(), 1);
        }
        return;
    }

    int Bayes::indicator(int x_id, int y_id)
    {
        int num = 0;
        for (int i = 0; i < y.size(); i++) {
            if (y[i] == y_id) {
                num += x[i][x_id];
            }
        }
        return num;
    }
    void Bayes::calculatePrior()
    {
        /* laplace smoothing */
        int lambda = 1;
        std::vector<int> y_num(labels.size());
        /* margin probability */
        for (int i = 0; i < margin.size(); i++) {
            int num = 0;
            for (int j = 0; j < x[0].size(); j++) {
                num += indicator(j, i);
            }
            margin[i] = double(num + lambda) / double(elementNum + 2 * lambda);
            y_num[i] = num;
        }
        /* conditional probability */
        for (int i = 0; i < condition.size(); i++) {
            int num = 0;
            for (int j = 0; j < condition[0].size(); j++) {
                num = indicator(j, i);
                condition[i][j] = double(num + lambda) / double(y_num[i] + lambda);
            }
        }
        return;
    }

    void Bayes::calculatePosterior(std::vector<int>& xi)
    {
        /* posterior probability */
        double p = 1;
        for (int i = 0; i < condition.size(); i++) {
            p = margin[i];
            for (int j = 0; j < condition[0].size(); j++) {
                if (xi[j] != 0) {
                    p *= condition[i][j];
                }
            }
            posterior[i] = p;
        }
        return;
    }

    std::string Bayes::classify(std::string& sample)
    {
        int index = 0;
        std::vector<std::string> strVect;
        std::vector<int> xi(elements.size(), 0);
        splitString(sample, strVect, " ");
        convertDataToFeature(strVect, xi);
        calculatePosterior(xi);
        double max_p = posterior[0];
        for (int i = 0; i < posterior.size(); i++) {
            if (max_p < posterior[i]) {
                max_p = posterior[i];
                index = i;
            }
        }
        return labels[index];
    }

    void Bayes::show()
    {
        /* show element */
        std::cout<<"show element:"<<std::endl;
        for (int i = 0; i < elements.size(); i++) {
            std::cout<<elements[i]<<" ";
        }
        std::cout<<std::endl;
        /* show label */
        std::cout<<"show label:"<<std::endl;
        for (int i = 0; i < labels.size(); i++) {
            std::cout<<labels[i]<<" ";
        }
        std::cout<<std::endl;
        /* show feature */
        std::cout<<"show feature:"<<std::endl;
        for (int i = 0; i < x.size(); i++) {
            for (int j = 0; j < x[i].size(); j++) {
                std::cout<<x[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
        /* show margin probability */
        std::cout<<"show margin probability:"<<std::endl;
        for (int i = 0; i < margin.size(); i++) {
            std::cout<<margin[i]<<std::endl;
        }
        std::cout<<"show conditional probability:"<<std::endl;
        for (int i = 0; i < condition.size(); i++) {
            for (int j = 0; j < condition[i].size(); j++) {
                std::cout<<condition[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
        return;
    }
}
