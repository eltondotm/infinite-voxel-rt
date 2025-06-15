#pragma once

#include <cuda_runtime.h>

/* Reads a .vox file into a texture buffer and populates the dimensions. */
void* loadVox(const char *filename, cudaExtent& dims) {
    FILE *fp = fopen(filename, "rb");

    if (!fp) {
        fprintf(stderr, "Error opening file '%s', errno=%d\n", filename, errno);
        return 0;
    }

    // Todo: extent to multiple objects
    size_t header_size = 60;
    const char *size_tag = "SIZE";
    const char *data_tag = "XYZI";

    char *header = (char *)malloc(header_size + 1);
    size_t read = fread(header, 1, header_size, fp);
    if (read != header_size) fprintf(stderr, "Error reading header");
    header[header_size] = '\0';

    const int *size_data = (int *)(strstr(header+20, size_tag)+12);
    dims = make_cudaExtent(size_data[0], size_data[2], size_data[1]);

    const int *data_loc = (int *)(strstr(header+44, data_tag)+12);
    int n_voxels = data_loc[0];
    free(header);

    uchar4 *voxels = (uchar4 *)malloc(n_voxels*sizeof(uchar4));
    read = fread(voxels, sizeof(uchar4), n_voxels, fp);

    // Todo: ensure 0 palette is not used
    void *data = calloc(dims.width * dims.height * dims.depth, sizeof(unsigned char));
    for (int i = 0; i < n_voxels; ++i) {
        uchar4 v = voxels[i];
        size_t idx = (v.x + dims.width*(v.z + dims.height*v.y));
        *((unsigned char *)data+idx) = v.w;
    }

    fclose(fp);

    return data;
}