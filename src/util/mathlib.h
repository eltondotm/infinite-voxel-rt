#pragma once

#include "helper_math.h"

#define EPS_F 0.0001f
#define PI_F 3.14159265358979323846264338327950288f

typedef unsigned int  uint;
typedef unsigned char uchar;

/////////////////////////////////////////////////
// Transformation matrices
/////////////////////////////////////////////////
typedef struct
{
    float4 m[3];
} float3x4;

__device__ float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

__device__ float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

/////////////////////////////////////////////////
// Device implementation of common functions
/////////////////////////////////////////////////

__device__ void swap(float& a, float& b)
{
    float temp = a;
    a = b;
    b = temp;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__device__ float4 rgbaIntToFloat(uint rgba)
{
    const uint mask = 0xFF;
    const float inv255 = 1.0f / 255.0f;
    uint r = rgba         & mask;
    uint g = (rgba >>  8) & mask;
    uint b = (rgba >> 16) & mask;
    uint a = (rgba >> 24) & mask;
    return make_float4((float)r * inv255, (float)g * inv255, (float)b * inv255, (float)a * inv255);
}