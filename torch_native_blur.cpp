/*
 * Simple toy app to help me get familiar with libtorch. Reads a png turns it 
 * into a torch::Tensor, blurs it and writes it back to disk.
 */

#include <ATen/core/TensorBody.h>
#include <ATen/core/jit_type.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/functional/conv.h>
#include <torch/nn/modules/conv.h>
#include <torch/serialize.h>
#include <torch/types.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <torch/torch.h>
#include <c10/util/ArrayRef.h>
#include <ATen/TypeDefault.h>

#include <vector>

#include <cstdio>
#include <cstdint>

using std::vector;

namespace F = torch::nn::functional;
using torch::Tensor;
using torch::TensorOptions;


void printSizes(const at::IntArrayRef& sizes) {
    size_t n = sizes.size();
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i == n - 1) { printf("%li", sizes[i]); }
        else            { printf("%li ", sizes[i]); }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s image.png image_out.png\n", argv[0]);
        exit(1);
    }

    int w, h, c;
    unsigned char *pixels = stbi_load(argv[1], &w, &h, &c, 0);
    if (!pixels) {
        fprintf(stderr, "Failed to read image at %s!\n", argv[1]);
        exit(1);
    }

    /* Format of pixels, from stb_image.h:
    The pixel data consists of *y scanlines of *x pixels,
    with each pixel consisting of N interleaved 8-bit components; the first
    pixel pointed to is top-left-most in the image. There is no padding between
    image scanlines or between pixels, regardless of format. The number of
    components N is 'desired_channels' if desired_channels is non-zero, or 
    *channels_in_file otherwise. 
    */

    // Keep in mind that this does not take ownership of the data!
    Tensor img = torch::from_blob(pixels, {h, w, c}, TensorOptions().dtype(torch::kUInt8));
    Tensor floatImg = img.to(TensorOptions().dtype(torch::kFloat)) / 255.0;

    Tensor gauss5 = torch::tensor({
        {0.03389831, 0.06779661, 0.08474576, 0.06779661, 0.03389831},
        {0.06779661, 0.15254237, 0.20338983, 0.15254237, 0.06779661},
        {0.08474576, 0.20338983, 0.25423729, 0.20338983, 0.08474576},
        {0.06779661, 0.15254237, 0.20338983, 0.15254237, 0.06779661},
        {0.03389831, 0.06779661, 0.08474576, 0.06779661, 0.03389831}});

    gauss5 /= gauss5.sum();
    
    // WHY is the shape here called sizes???
    int k = gauss5.sizes()[0];

    // Recall that sizes must be:
    // Input: (batch_size, in_channels , height, width)
    // Kernel: (out_channels, in_channels/groups, kernel_height, kernel_width)

    floatImg = floatImg.movedim(-1, 0).unsqueeze(0);

    gauss5 = gauss5.unsqueeze(0);
    vector<Tensor> kernels(c, gauss5);
    gauss5 = torch::stack(kernels);

    // printf("Image size: ");  printSizes(floatImg.sizes()); printf("\n");
    // printf("Kernel size: "); printSizes(gauss5.sizes());   printf("\n");

    // For whatever reason the stride is set to {1, 1} by default, which does
    // not match the dims of floatImg and gauss5.
    Tensor blurredImg = F::conv2d(floatImg, gauss5, F::Conv2dFuncOptions().stride(1).groups(c));

    blurredImg = (blurredImg * 255.0).to(TensorOptions().dtype(torch::kUInt8));
    blurredImg = blurredImg.squeeze();
    blurredImg = blurredImg.movedim(0, -1);
    printf("Image size: ");  printSizes(blurredImg.sizes()); printf("\n");
    torch::save(blurredImg, "out.pt");

    // FIXME Writing out the PNG does not work yet, check again the data format!
    const unsigned char* outPixels = static_cast<const unsigned char*>(blurredImg.data_ptr());
    int stride = w * c;
    int success = stbi_write_png(argv[2], w, h, c, outPixels, stride);
    if (!success) {
        fprintf(stderr, "Failed to write %s!\n", argv[2]);
        exit(1);
    } else {
        printf("Wrote: %s\n", argv[2]);
    }

    // To save as a torch tensor simply invoke:
    // torch::save(blurredImg, "my_tensor.pt");
    // Note that this saves a module and not a tensor. It can be loaded in 
    // Python with:
    // tensors_model = torch.load("build/my_tensor.pt")
    // my_tensor = list(tensors_model.parameters())[0]
    // Further tensors would be at index 1, 2, 3, etc

    stbi_image_free(pixels);

    return 0;
} 