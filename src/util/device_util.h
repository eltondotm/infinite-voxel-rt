#pragma once

template <typename T>
inline __device__ void swap(T& l, T& r) {
    T temp = l;
    l = r;
    r = temp;
}

inline __device__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

inline __device__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}

inline __device__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}

inline __device__ float length(float u, float v)
{
    return sqrtf(u*u + v*v);
}
