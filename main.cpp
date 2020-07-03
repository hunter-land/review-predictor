#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

//typedef std::vector<bool> reviewFeatureVector;
struct review {
	std::string text;
	std::vector<bool> features;
	bool classlabel;
};
struct preprocessedDataStruct {
	std::vector<std::string> vocabulary;
	std::vector<review> reviews;
};
std::ostream& operator<<(std::ostream& os, const preprocessedDataStruct& ppds) {
	for(size_t i = 0; i < ppds.vocabulary.size(); i++) {
		//if(i < 5) {
		//	std::cout << "Word (\"" << ppds.vocabulary[i] << "\")has length of " << ppds.vocabulary[i].length() << std::endl;
		//}
		os << ppds.vocabulary[i] << ",";
	}
	os << "classlabel" << std::endl;

	for(size_t i = 0; i < ppds.reviews.size(); i++) {
		for(size_t j = 0; j < ppds.reviews[i].features.size() - 1; j++) {
			os << (int)ppds.reviews[i].features[j] << ",";
		}
		os << (int)ppds.reviews[i].features[ppds.reviews[i].features.size()-1] << std::endl;
	}
	return os;
}
//The classifier, which can be trained on any preprocessed data, will predict a review's classlabel using its features.
//Note: It does not check if reviews use different vocab, so filter that by hand.
class classifier {
private:
	std::vector<std::string> m_vocabulary;
	std::vector<double> m_probPosi; //Probability a word is for a positive review
	std::vector<double> m_probNega; //Prob a word is for negative
public:
	classifier(preprocessedDataStruct ppds) {
		m_vocabulary = ppds.vocabulary;
		m_probPosi.resize(m_vocabulary.size());
		m_probNega.resize(m_vocabulary.size());

		int posiWords = 0;
		int negaWords = 0;
		int posiReviews = 0;
		int negaReviews = 0;

		for(review &r : ppds.reviews) {
			if(r.classlabel) {
				posiReviews++;
				for(int i = 0; i < r.features.size()-1; i++) {
					if(r.features[i]) { //If the word is present
						m_probPosi[i]++;
						posiWords++;
					}
				}
			} else {
				negaReviews++;
				for(int i = 0; i < r.features.size()-1; i++) {
					if(r.features[i]) { //If the word is present
						m_probNega[i]++;
						negaWords++;
					}
				}
			}
		}

		std::cout << "Built classifier from " << ppds.reviews.size() << " reviews, " << posiWords << " positive words, and " << negaWords << " negative words." << std::endl;
		for(double &d : m_probPosi) {
			d/=posiReviews;
		}
		for(double &d : m_probNega) {
			d/=negaReviews;
		}
	}
	
	bool predictReview(review r) {
		//Asume review is negative, what are the odds all words are present?
		//P(F[i] = 1 | classlabel = 0) * 
		double probNegative = 1;
		double probPositive = 1;
		for(int i = 0; i < r.features.size()-1; i++) {
			if(r.features[i]) { //Word is present
				probPositive *= m_probPosi[i];
				probNegative *= m_probNega[i];
			}
		}

		//std::cout << "Probability for positive:\t" << probPositive << std::endl;
		//std::cout << "Probability for negative:\t" << probNegative << std::endl;
		return probPositive > probNegative; //If it is more likely to be a positive review than negative, return true (1)
	}
};

preprocessedDataStruct preprocess(std::string filename, std::vector<std::string> *vocab = nullptr); //Passing a vocab will force use that vocab
std::string cleanWord(std::string s); //Remove punctuation, make all lower case, etc

int main() {
	preprocessedDataStruct trainingDataSet = preprocess("./trainingSet.txt");
	preprocessedDataStruct testDataSet = preprocess("./testSet.txt", &trainingDataSet.vocabulary);

	std::ofstream pptr("./preprocessed_train.txt");
	pptr << trainingDataSet;
	pptr.close();
	std::ofstream ppte("./preprocessed_test.txt");
	ppte << testDataSet;
	ppte.close();

	classifier c(trainingDataSet);
	//review r = trainingDataSet.reviews[0];
	//std::cout << "\"" << r.text << "\"" << std::endl << "\tPredicted: " << c.predictReview(r) << "\tActual: " << r.classlabel << std::endl;
	
	
	int accurateCount = 0;
	int positives = 0;
	int negatives = 0;
	for(review &r : trainingDataSet.reviews) {
		bool p = c.predictReview(r);
		//std::cout << "Predicted " << (int)p << " and was actually " << (int)r.classlabel << std::endl;
		if(p == r.classlabel) {
			accurateCount++;
		}
		if(r.classlabel) {
			positives++;
		} else {
			negatives++;
		}
	}
	std::cout << "Training set sanity check:" << std::endl;
	std::cout << "\tCorrectly assumed classlabel for " << accurateCount << " of " << trainingDataSet.reviews.size() << " (" << (double)accurateCount/trainingDataSet.reviews.size() << "%) reviews." << std::endl;
	std::cout << "\tThere were " << positives << " positive reviews and " << negatives << " negative reviews." << std::endl;
	std::cout << std::endl;

	accurateCount = 0;
	positives = 0;
	negatives = 0;
	for(review &r : testDataSet.reviews) {
		bool p = c.predictReview(r);
		if(p == r.classlabel) {
			accurateCount++;
		}
		if(r.classlabel) {
			positives++;
		} else {
			negatives++;
		}
	}
	std::cout << "Testing set results:" << std::endl;
	std::cout << "\tCorrectly assumed classlabel for " << accurateCount << " of " << testDataSet.reviews.size() << " (" << (double)accurateCount/testDataSet.reviews.size() << "%) reviews." << std::endl;
	std::cout << "\tThere were " << positives << " positive reviews and " << negatives << " negative reviews." << std::endl;
	std::cout << std::endl;

	std::ofstream rout("./results.txt");
	rout << "Trained on \"./trainingSet.txt\", tested on \"./testingSet.txt\":" << std::endl;
	rout << "\tCorrectly assumed classlabel for " << accurateCount << " of " << testDataSet.reviews.size() << " (" << (double)accurateCount/testDataSet.reviews.size() << "%) reviews." << std::endl;
	rout.close();
}



preprocessedDataStruct preprocess(std::string filename, std::vector<std::string> *vocab) {
	preprocessedDataStruct set;
	if(vocab != nullptr) {
		set.vocabulary = *vocab;
	}
	std::ifstream fin(filename);
	std::string line;
	//Get the reviews and words from them, but don't populate set.reviewFeatures yet!
	while(std::getline(fin, line)) {
		//reviews.push_back(line);

		review r;
		r.classlabel = std::atoi(line.substr(line.find('\t')+2, 1).data());
		//std::cout << line << std::endl;
		//std::cout << line.substr(line.find('\t')+2) << std::endl;
		//std::cout << (int)r.classlabel << std::endl;
		line = line.substr(0, line.find('\t'));
		r.text = line;
		
		std::string word;
		while(line.find(' ') != std::string::npos) { //For each word in the review
			word = line.substr(0, line.find(' '));
			line = line.substr(word.length() + 1);
			word = cleanWord(word);
			if(vocab == nullptr && word.length() > 0 && std::find(set.vocabulary.begin(), set.vocabulary.end(), word) == set.vocabulary.end()) { //New word, so we add it
				set.vocabulary.push_back(word);
			}
		}
		set.reviews.push_back(r);
	}
	fin.close();
	
	//Sort vocab list
	std::sort(set.vocabulary.begin(), set.vocabulary.end());
	//Now populate reviewFeatures vector
	for(review &r : set.reviews) {
		//review r;
		r.features.resize(set.vocabulary.size() + 1);
		//s = s.substr(0, s.find('\t'));
		//r.text = s;

		std::string s = r.text;
		std::string word;
		while(s.find(' ') != std::string::npos) { //For each word in the review
			word = s.substr(0, s.find(' '));
			s = s.substr(word.length() + 1);
			word = cleanWord(word);

			r.features[std::distance(set.vocabulary.begin(), std::find(set.vocabulary.begin(), set.vocabulary.end(), word))] = true;
		}
		r.features[set.vocabulary.size()] = r.classlabel; //This is done here because line 212 above will make this true when a word is not in the vocab, so rather than an if statement I just overwrite it here
		//set.reviews.push_back(r);
	}
	
	// std::cout << "Returning a set with " << set.vocabulary.size() << " vocab entries and " << set.reviewFeatures.size() << " reviews." << std::endl;
	// for(int i = 0; i < 3; i++) {
	// 	std::cout << '\t' << set.vocabulary[i] << std::endl;
	// }
	// for(int i = 0; i < 10; i++) {
	// 	std::cout << "\t\t" << (int)set.reviewFeatures[0][i] << std::endl;
	// }
	return set;
}
std::string cleanWord(std::string s) {
	//std::cout << "Cleaning word \"" << s << "\" to ";
	size_t pos;
	for(char c : {'.',',','!','\'','"','(',')','-'}) {
		while((pos = s.find(c)) != std::string::npos) {
			s.erase(pos);
		}
	}
	for(int i = 0; i < s.length(); i++) {
		s[i] = tolower(s[i]);
	}
	//std::cout << s << std::endl;
	return s;
}
