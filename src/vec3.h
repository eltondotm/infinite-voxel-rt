#pragma once

#include <math.h>
#include <stdlib.h>
#include <ostream>

struct Vec3 {
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ explicit Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    __host__ __device__ explicit Vec3(float f) : x(f), y(f), z(f) {}

    __host__ __device__ float& operator[](int idx) { return data[idx]; }
    __host__ __device__ float operator[](int idx) const { return data[idx]; }

    __host__ __device__ Vec3 operator+=(const Vec3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    __host__ __device__ Vec3 operator-=(const Vec3 &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    __host__ __device__ Vec3 operator*=(const Vec3 &v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    __host__ __device__ Vec3 operator/=(const Vec3 &v) {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }

    __host__ __device__ Vec3 operator+=(const float s) {
        x += s;
        y += s;
        z += s;
        return *this;
    }
    __host__ __device__ Vec3 operator-=(const float s) {
        x -= s;
        y -= s;
        z -= s;
        return *this;
    }
    __host__ __device__ Vec3 operator*=(const float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    __host__ __device__ Vec3 operator/=(const float s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    __host__ __device__ Vec3 operator+(const Vec3 &v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    __host__ __device__ Vec3 operator-(const Vec3 &v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    __host__ __device__ Vec3 operator*(const Vec3 &v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }
    __host__ __device__ Vec3 operator/(const Vec3 &v) const {
        return Vec3(x / v.x, y / v.y, z / v.z);
    }

    __host__ __device__ Vec3 operator+(const float s) const {
        return Vec3(x + s, y + s, z + s);
    }
    __host__ __device__ Vec3 operator-(const float s) const {
        return Vec3(x - s, y - s, z - s);
    }
    __host__ __device__ Vec3 operator*(const float s) const {
        return Vec3(x * s, y * s, z * s);
    }
    __host__ __device__ Vec3 operator/(const float s) const {
        return Vec3(x / s, y / s, z / s);
    }

    __host__ __device__ bool operator==(const Vec3 &v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    __host__ __device__ bool operator!=(const Vec3 &v) const {
        return x != v.x || y != v.y || z != v.z;
    }

    __host__ __device__ Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }
    __host__ __device__ bool valid() const {
        return !(isinf(x) || isinf(y) || isinf(z) || isnan(x) ||
                 isnan(y) || isnan(z));
    }

    __host__ __device__ Vec3 normalize() {
        float n = length();
        x /= n;
        y /= n;
        z /= n;
        return *this;
    }
    __host__ __device__ Vec3 unit() const {
        float n = length();
        return Vec3(x / n, y / n, z / n);
    }
    __host__ __device__ float length_squared() const {
        return x * x + y * y + z * z;
    }
    __host__ __device__ float length() const {
        return sqrtf(length_squared());
    }
    __host__ __device__ Vec3 clamp() {
        x = fminf(1, fmaxf(0, x));
        y = fminf(1, fmaxf(0, y));
        z = fminf(1, fmaxf(0, z));
        return *this;
    }

    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[3] = {};
    };
};

__host__ __device__ inline Vec3 operator+(const float s, const Vec3 &v) {
    return Vec3(v.x + s, v.y + s, v.z + s);
}
__host__ __device__ inline Vec3 operator-(const float s, const Vec3 &v) {
    return Vec3(v.x - s, v.y - s, v.z - s);
}
__host__ __device__ inline Vec3 operator*(const float s, const Vec3 &v) {
    return Vec3(v.x * s, v.y * s, v.z * s);
}
__host__ __device__ inline Vec3 operator/(const float s, const Vec3 &v) {
    return Vec3(s / v.x, s / v.y, s / v.z);
}
__host__ __device__ inline float dot(const Vec3 &l, const Vec3 &r) {
    return l.x * r.x + l.y * r.y + l.z * r.z;
}
__host__ __device__ inline Vec3 cross(const Vec3 &l, const Vec3 &r) {
    return Vec3(l.y * r.z - l.z * r.y, l.z * r.x - l.x * r.z, l.x * r.y - l.y * r.x);
}
__host__ __device__ inline Vec3 reflect(const Vec3 &i, const Vec3 &n) {
    return i - 2.0f * n * dot(n,i);
}
__host__ __device__ inline Vec3 hmin(const Vec3 &l, const Vec3 &r) {
    float x = fminf(l.x, r.x);
    float y = fminf(l.y, r.y);
    float z = fminf(l.z, r.z);
    return Vec3(x, y, z);
}
__host__ __device__ inline Vec3 hmax(const Vec3 &l, const Vec3 &r) {
    float x = fmaxf(l.x, r.x);
    float y = fmaxf(l.y, r.y);
    float z = fmaxf(l.z, r.z);
    return Vec3(x, y, z);
}

inline std::ostream& operator<<(std::ostream& out, const Vec3 &v) {
    out << "{" << v.x << "," << v.y << "," << v.z << "}";
    return out;
}

__host__ __device__ inline bool operator<(const Vec3 &l, const Vec3 &r) {
    if(l.x == r.x) {
        if(l.y == r.y) {
            return l.z < r.z;
        }
        return l.y < r.y;
    }
    return l.x < r.x;
}
