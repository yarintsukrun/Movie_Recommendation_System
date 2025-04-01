//
// Created by ירין on 29/07/2024.
//


#include "User.h"
#include "RecommendationSystem.h"
#include <utility>

User::User(std::string& username,
    rank_map map, std::shared_ptr<RecommendationSystem> shared):
_username(username), _user_ranked_movie_list(std::move(map)),
_shared_recommand_system(std::move(shared))
{}


std::string User::get_name() const{return _username;}

void User::add_movie_to_user(const std::string &name, int year,
    const std::vector<double> &features, double rate) {
    _user_ranked_movie_list
    [_shared_recommand_system->add_movie_to_rs(name,year, features)]
    = rate;
}

rank_map User::get_ranks() const{
    return _user_ranked_movie_list;
}

sp_movie User::get_rs_recommendation_by_content() const {
    return _shared_recommand_system->recommend_by_content(*this);
}


sp_movie User::get_rs_recommendation_by_cf(int k) const {
    return _shared_recommand_system->recommend_by_cf(*this,k);

}

double User::get_rs_prediction_score_for_movie
(const std::string& name, int year, int k) const {
    auto movie = _shared_recommand_system->get_movie(name,year);
    if (movie == nullptr) {
        throw(std::invalid_argument){"this movie doesnt exist in shared"};
    }
    return _shared_recommand_system->predict_movie_score(*this,movie,k);
}


std::ostream& operator<<(std::ostream& os, User& user) {
    os << "name: "<< user.get_name() << std::endl;
    os << *(user._shared_recommand_system) << std::endl;
    return os;
}