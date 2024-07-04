#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

//Assuming an appropriate edge detection algorithm was applied on the image
// Hough Transform to detect lines
std::vector<std::pair<int, int>> houghTransform(const std::vector<std::vector<int>>& edges, int width, int height, int threshold) {
    int maxDist = std::sqrt(width * width + height * height);
    std::vector<std::vector<int>> accumulator(2 * maxDist, std::vector<int>(180, 0));
    std::vector<std::pair<int, int>> lines;

    // Voting in Hough space
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (edges[y][x] > 128) {  // Edge threshold
                for (int theta = 0; theta < 180; ++theta) {
                    double radians = theta * M_PI / 180.0;
                    int rho = static_cast<int>(x * std::cos(radians) + y * std::sin(radians));
                    int rhoIdx = rho + maxDist;
                    if (rhoIdx >= 0 && rhoIdx < 2 * maxDist) {
                        accumulator[rhoIdx][theta]++;
                    }
                }
            }
        }
    }

    // Extract lines from the accumulator
    for (int rhoIdx = 0; rhoIdx < 2 * maxDist; ++rhoIdx) {
        for (int theta = 0; theta < 180; ++theta) {
            if (accumulator[rhoIdx][theta] > threshold) {
                int rho = rhoIdx - maxDist;
                lines.emplace_back(rho, theta); // Store detected line
            }
        }
    }

    return lines;
}