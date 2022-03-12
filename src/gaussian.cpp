#include "../inc/gaussian.hpp"
#include <math.h>

#define PI 3.14159265359
#define E 2.71828

float* GaussianArgs::create_kernel()
{
    float* gaussian = new float[SIZE];
    for(int gx = 0; gx < DIAMETER; gx++) {
        for(int gy = 0; gy < DIAMETER; gy++) {
            int m_gx = gx - RADIUS;
            int m_gy = gy - RADIUS;
            float dev2 = DEVIATION * DEVIATION;
            int m_gx2 = m_gx * m_gx;
            int m_gy2 = m_gy * m_gy;
            double exp = (m_gx2 + m_gy2) / (2 * dev2);
            double frac = 1 / (2 * PI * dev2);
            double ans = frac * pow(E, -1 * exp);
            gaussian[(DIAMETER * gy) + gx] = ans;
        }
    }
    return gaussian;
}