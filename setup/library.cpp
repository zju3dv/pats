
#include <iostream>
#include <torch/torch.h>
#include "torch/script.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <iostream>
#include <string>
#include <memory>
using namespace torch::indexing;
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

std::vector<at::Tensor> lltm_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell) {
    auto X = torch::cat({old_h, input}, /*dim=*/1);

    auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
    auto gates = gate_weights.chunk(3, /*dim=*/1);

    auto input_gate = torch::sigmoid(gates[0]);
    auto output_gate = torch::sigmoid(gates[1]);
    auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

    auto new_cell = old_cell + candidate_cell * input_gate;
    auto new_h = torch::tanh(new_cell) * output_gate;

    return {new_h,
            new_cell,
            input_gate,
            output_gate,
            candidate_cell,
            X,
            gate_weights};
}

torch::Tensor resize(const torch::Tensor& input_tensor, const torch::Tensor& bound) {
    auto input_shape = input_tensor.sizes();
    auto bound_shape = bound.sizes();
    long patch_shape_x = 32;
    long patch_shape_y = 32;
    torch::Tensor output = torch::zeros({bound_shape[0], input_shape[1], patch_shape_y * 3, patch_shape_x * 3}, torch::kFloat).to(input_tensor.device());
    for(int i = 0; i < bound_shape[0]; i++)
    {
        long sequence = bound.index({i, 4}).item().toLong();
        auto input_patch = input_tensor.index({sequence / 10000, "...", "...", "..."})
                .narrow(1, bound.index({i, 0}).item().toLong(),
                        (bound.index({i, 1}) - bound.index({i, 0})).item().toLong()).narrow(2, bound.index({i, 2}).item().toLong(),
                                (bound.index({i, 3}) - bound.index({i, 2}) + 1).item().toLong()).unsqueeze(0);
        torch::Tensor resized_map = torch::upsample_bilinear2d(input_patch, {patch_shape_y * 3, patch_shape_x * 3}, true);
        output.index_put_({i,"...", "...", "..."}, resized_map);
    }
//    std::cout << output.device() << std::endl;
//    std::cout << "test" << std::endl;
    return output;
}

//torch::Tensor resize(const torch::Tensor& input_tensor, const torch::Tensor& bound) {
//    auto input_shape = input_tensor.sizes();
//    auto bound_shape = bound.sizes();
//    long patch_shape_x = input_shape[2] / 15;
//    long patch_shape_y = input_shape[3] / 20;
//    torch::Tensor output = torch::zeros(input_shape, torch::kFloat);
//    for(int i = 0; i < bound_shape[0]; i++)
//    {
//        for(int j = 0; j < bound_shape[1]; j++)
//        {
//            auto input_patch = input_tensor.index({i, "...", "...", "..."})
//                    .narrow(1, bound.index({i, j, 0}).item().toLong(),
//                            (bound.index({i, j, 1}) - bound.index({i, j, 0})).item().toLong())
//                    .narrow(2, bound.index({i, j, 2}).item().toLong(),
//                            (bound.index({i, j, 3}) - bound.index({i, j, 2}) + 1).item().toLong()).unsqueeze(0);
//            torch::Tensor resized_map = torch::upsample_bilinear2d(input_patch, {patch_shape_y, patch_shape_x}, true);
//            output.index_put_({i,"...", Slice((j / 20) * patch_shape_y, (j / 20 + 1) * patch_shape_y),
//                               Slice((j % 20) * patch_shape_x, (j % 20 + 1) * patch_shape_x)}, resized_map.squeeze(0));
//        }
//    }
//    return input_tensor;
//}


PYBIND11_MODULE(tensor_resize, m) {
    m.def("tensor_resize", &resize, "feature resize");
//    m.def("tensor_resize", &lltm_forward, "feature resize");
}
