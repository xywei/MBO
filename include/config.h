//
// Created by Xiaoyu Wei on 8/5/2016.
//

#ifndef MBOX_CONFIG_H
#define MBOX_CONFIG_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <type_traits>

#include <deal.II/base/logstream.h>
#include <deal.II/base/conditional_ostream.h>

/**
 * Existence test
 * @brief A simple inline function that checks whether a file exists
 */
inline bool exists_test (const std::string & name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

#endif // MBOX_CONFIG_H
