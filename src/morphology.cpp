#include "../inc/morphology.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PADDING 1



bool* MorphSize::element_cross()
{
    bool* element = new bool[SIZE];
    for(int x = 0; x < DIAMETER; x++) {
        for(int y = 0; y < DIAMETER; y++) {
            int idx = y + DIAMETER * x;
            bool cond1 = x > RADIUS - PADDING && x < RADIUS + PADDING;
            bool cond2 = y > RADIUS - PADDING && y < RADIUS + PADDING;
            if(cond1 || cond2) element[idx] = true;
            else element[idx] = false;
        }
    }
    return element;
}

bool* MorphSize::element_circle()
{
    bool* element = new bool[SIZE];
    for(int x = 0, cx = -RADIUS; x < DIAMETER; x++, cx++) {
        for(int y = 0, cy = -RADIUS; x < DIAMETER; x++, cy++) {
            int tid = y * DIAMETER + x;
            bool cond = pow(cx, 2) + pow(cy, 2) <= pow(RADIUS, 2);
            if(cond) element[tid] = true;
            else element[tid] = false;
        }
    }
    return element;
}

 bool* MorphSize::element_recta()
{
    bool* element = new bool[SIZE];
    for(int x = 0; x < DIAMETER; x++) {
        for(int y = 0; y < DIAMETER; y++) {
            int idx = y + DIAMETER * x;
            element[idx] = true;
        }
    }
    return element;
}
