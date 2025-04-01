//
// Created by ירין on 29/07/2024.
//
#include "Movie.h"
#include "User.h"
#include "RecommendationSystem.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "RecommendationSystemLoader.h"

double scalar_mult(const std::vector<double>
    &lhs, const std::vector<double> &rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("wrong size vectors");
    }

    double result = 0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

double standard_norm(const std::vector<double> &vec) {
    double result = 0;
    for (double num : vec) {
        result += std::pow(num, 2);
    }
    return std::sqrt(result);
}

double similarity(const std::vector<double> &lhs,
    const std::vector<double> &rhs) {
    double lhs_norm = standard_norm(lhs);
    double rhs_norm = standard_norm(rhs);

    if (lhs_norm == 0 || rhs_norm == 0) {
        return 0; // Handle the case where one or both vectors have zero norm
    }

    return scalar_mult(lhs, rhs) / (lhs_norm * rhs_norm);
}
std::vector<double> norm(const std::vector<double> &vec){
    std::vector<double> newvec = vec;
    for ( auto &num : newvec) {
        if ( num < 0) {
            num = 0;
        }
    }
    return newvec;
}


std::vector<double> RecommendationSystem::make_self_norm_vec
(const User& user) {
    rank_map copymap = user.get_ranks();
    double sum = 0;
    for (const auto& num : copymap) {
        sum += num.second;
    }
    double factorial = sum / static_cast<double>(copymap.size());
    for (auto& num : copymap) {
        num.second -= factorial;
    }
    std::vector<double> vec(_movie_features.begin()->second.size(), 0);
    for (const auto& name : copymap) {
        auto it = _movie_features[name.first];
            for (size_t i = 0; i < vec.size(); ++i) {
                vec[i] += it[i] * name.second;
            }
        }
    return vec;
}



sp_movie RecommendationSystem::recommend_by_content(const User& user) {
    rank_map copymap = user.get_ranks();
    std::vector<double> vec = make_self_norm_vec(user);

    double best_result = -std::numeric_limits<double>::infinity();
    sp_movie best_movie;

    for (const auto& spmovie : _movie_features) {
        auto it = copymap.find(spmovie.first);
        if (it == copymap.end()) {
            std::vector<double> finded_vec = spmovie.second;
            double try_result = similarity(vec, finded_vec);
            if (try_result > best_result) {
                best_result = try_result;
                best_movie = spmovie.first;
            }
        }
    }
    return best_movie;
}


double RecommendationSystem::predict_movie_score
(const User& user, const sp_movie& movie, int k) {
    auto movie_it = _movie_features.find(movie);
    if (movie_it == _movie_features.end()) {
        throw std::runtime_error("Movie not found in the feature map.");
    }

    std::vector<double> input_movie_vec = movie_it->second;

    std::map<double, sp_movie, std::greater<double>> ranked_by_user;

    for (const auto& users_movie : user.get_ranks()) {
        auto features_it = _movie_features.find(users_movie.first);
        if (features_it != _movie_features.end()) {
            std::vector<double> check = features_it->second;
            double sim = similarity(input_movie_vec, check);
            ranked_by_user[sim] = users_movie.first;
        }
    }

    double weighted_sum = 0.0;
    double similarity_sum = 0.0;

    int count = 0;
    for (const auto& item : ranked_by_user) {
        if (count >= k){ break;}
        auto rank_it = user.get_ranks()[item.second];
            weighted_sum += item.first * rank_it;
            similarity_sum += item.first;
            ++count;
    }
    if (similarity_sum == 0) {
        return 0;
    }

    return weighted_sum / similarity_sum;
}



sp_movie RecommendationSystem::recommend_by_cf(const User& user, int k) {
    std::map<double,sp_movie> resultmap;

    for (const auto& sharedmovie: _movie_features) {
        auto features_it = user.get_ranks().find(sharedmovie.first);
        if (features_it == user.get_ranks().end()) {
            resultmap[predict_movie_score(user, sharedmovie.first, k)] =
                sharedmovie.first;
        }
    }
    auto ptr = resultmap.end();
    --ptr;
    return ptr->second;

}

sp_movie RecommendationSystem::add_movie_to_rs
(const std::string& name,int year,const std::vector<double>& features) {
    auto spmovie = std::make_shared<Movie>(name,year);
    _movie_features[spmovie] = features;
    return spmovie;

}

sp_movie RecommendationSystem::get_movie(const std::string& name, int year) {
    auto spmovie = std::make_shared<Movie>(name, year);
    auto it = _movie_features.find(spmovie);
    if (it == _movie_features.end()) {
        return nullptr;
    }
    return it->first;
}


std::ostream& operator<<(std::ostream& os, RecommendationSystem &sys) {
    std::vector<sp_movie> movies;
    for (const auto& entry  : sys._movie_features) {
        movies.push_back(entry.first);
    }

    // Step 2: Sort the vector using the < operator of Movie
    std::sort(movies.begin(), movies.end(),
        [](const sp_movie& lhs, const sp_movie& rhs) {
       return *lhs < *rhs;
   });

    // Step 3: Print the sorted movies
    for (const auto& movie : movies) {
        os << *movie;
    }

    os << std::endl;
    return os;
}