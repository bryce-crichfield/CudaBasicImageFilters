#pragma once

typedef struct GaussianArgs {
    int DIAMETER;
    int RADIUS;
    int SIZE;
    float DEVIATION;
    int BYTES;
    int TILE_W;

    GaussianArgs(int diameter, float deviation)
    :   DIAMETER(diameter),
        RADIUS((diameter - 1) / 2),
        SIZE(diameter*diameter),
        DEVIATION(deviation),
        BYTES(SIZE * sizeof(float)),
        TILE_W((RADIUS - 1) / 2)
    { }

    float* create_kernel();

} GaussianArgs;