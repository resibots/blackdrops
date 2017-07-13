#ifndef UTILS_DART_UTILS_HPP
#define UTILS_DART_UTILS_HPP

#include <dart/dart.hpp>
#include <dart/utils/SkelParser.hpp>
#include <fstream>
#include <streambuf>
#include <string>

namespace utils {
    inline dart::dynamics::SkeletonPtr load_skel(const std::string& filename, const std::string& robot_name)
    {
        // Load file into string
        std::ifstream t(filename);
        std::string str((std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>());

        dart::simulation::WorldPtr world = dart::utils::SkelParser::readWorldXML(str);

        return world->getSkeleton(robot_name);
    }
}

#endif