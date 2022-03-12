#pragma once

#include "math.h"

#define OPEN_CLOSE 0
#define CLOSE_OPEN 1

namespace ELEMENT {
    static const int CROSS = 0;
    static const int CIRCLE = 1;
    static const int RECTANGLE = 2;
}

typedef struct MorphSize {
    int DIAMETER;
    int RADIUS;
    int SIZE;
    int BYTES;

    MorphSize(int power) 
    :   DIAMETER (pow(2, power) + 1),
        RADIUS ((DIAMETER - 1) / 2),
        SIZE  (DIAMETER*DIAMETER),
        BYTES (SIZE * sizeof(bool))
    {}

    bool* element_cross();
    bool* element_circle();
    bool* element_recta();
} MorphSize;