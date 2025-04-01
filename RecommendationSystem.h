//
// Created on 2/20/2022.
//

#ifndef RECOMMENDATIONSYSTEM_H
#define RECOMMENDATIONSYSTEM_H
#include "User.h"
#include "Movie.h"
#include "RecommendationSystemLoader.h"



typedef std::unordered_map<sp_movie, std::vector<double>,
hash_func,equal_func> movie_features;

class RecommendationSystem {
private:
	movie_features _movie_features;

public:


	//explicit RecommendationSystem()
	RecommendationSystem() : _movie_features(1,sp_movie_hash,
		sp_movie_equal){}
    /**
     * adds a new movie to the system
     * @param name name of movie
     * @param year year it was made
     * @param features features for movie
     * @return shared pointer for movie in system
     */
	sp_movie add_movie_to_rs(const std::string& name,int year,
		const std::vector<double>& features);


    /**
     * a function that calculates the movie with highest score based
     * on movie features
     * @param ranks user ranking to use for algorithm
     * @return shared pointer to movie in system
     */
	sp_movie recommend_by_content(const User& user);

    /**
     * a function that calculates the movie with highest
     * predicted score based on ranking of other movies
     * @param ranks user ranking to use for algorithm
     * @param k
     * @return shared pointer to movie in system
     */
	sp_movie recommend_by_cf(const User& user, int k);


    /**
     * Predict a user rating for a movie given argument using
     * item cf procedure with k most similar movies.
     * @param user_rankings: ranking to use
     * @param movie: movie to predict
     * @param k:
     * @return score based on algorithm as described in pdf
     */
	double predict_movie_score(const User &user, const sp_movie &movie,
												  int k);

	/**
	 * gets a shared pointer to movie in system
	 * @param name name of movie
	 * @param year year movie was made
	 * @return shared pointer to movie in system
	 */
	sp_movie get_movie(const std::string& name,int year);

	std::vector<double> make_self_norm_vec(const User& user);

	friend std::ostream& operator<<(std::ostream& os,
		RecommendationSystem &sys);

};

double scalar_mult(const std::vector<double> &lhs,
	const std::vector<double> &rhs);
double standard_norm(const std::vector<double> &vec);
double similarity(const std::vector<double> &lhs,
	const std::vector<double> &rhs);

#endif //RECOMMENDATIONSYSTEM_H
