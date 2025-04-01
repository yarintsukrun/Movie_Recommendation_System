
#ifndef INC_23B_C_C__EX5_MOVIE_H
#define INC_23B_C_C__EX5_MOVIE_H

#include <iostream>
#include <vector>
#include <memory>
#define HASH_START 17



class Movie;

typedef std::shared_ptr<Movie> sp_movie;

/**
 * those declartions and typedefs are given to y
 * ou and should be used in the ex
 */
typedef std::size_t (*hash_func)(const sp_movie& movie);
typedef bool (*equal_func)(const sp_movie& m1,const sp_movie& m2);
std::size_t sp_movie_hash(const sp_movie& movie);
bool sp_movie_equal(const sp_movie& m1,const sp_movie& m2);

class Movie
{
private:
 std::string _name;
 int _year;
   
public:
    /**
     * constructor
     * @param name: name of movie
     * @param year: year it was made
     */
    Movie(std::string  name, size_t year);

    /**
     * returns the name of the movie
     * @return const ref to name of movie
     */
    std::string get_name() const;

    /**
     * returns the year the movie was made
     * @return year movie was made
     */
    int get_year() const;

	/**
     * operator< for two movies
     * @param rhs: right hand side
     * @param lhs: left hand side
     * @return returns true if (lhs.year) < rhs.year or (
     * rhs.year == lhs.year & lhs.name < rhs.name) else return false
     */
    bool operator<(const Movie& rhs) const;

    /**
     * operator<< for movie
     * @param os ostream to output info with
     * @param movie movie to output
     */
 friend std::ostream& operator<<(std::ostream& os,const Movie& sp_movie);
};


#endif //INC_23B_C_C__EX5_MOVIE_H
