#pragma once

template <typename T>
__device__ void swap(T& l, T& r) {
    T temp = l;
    l = r;
    r = temp;
}
