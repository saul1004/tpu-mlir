//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1688.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace llvm;
using namespace tpu_mlir::backend;

namespace tpu_mlir {

namespace bm168x {

class GRUGlobalBuffer : public OpRewritePattern<tpu::GRUOp> {
public:
  using OpRewritePattern<tpu::GRUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::GRUOp GRUOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(GRUOp.getBuffer())) {
      return failure();
    }
    auto attr = GRUOp.parseParam();
    auto type = module::getStorageType(GRUOp.getInput());
    // add buffer
    std::vector<int64_t> buffer_shape = {5, attr.batch_size, attr.hidden_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(GRUOp, buffer_type);
    GRUOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == GRUOp.getBuffer() && operand.getOwner() == GRUOp;
    });
    return success();
  }
};

class FAttentionGlobalBuffer : public OpRewritePattern<tpu::FAttentionOp> {
public:
  using OpRewritePattern<tpu::FAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::FAttentionOp FaOp,
                                PatternRewriter &rewriter) const override {
    // if (!module::isNone(FaOp.getBuffer())) {
    //   return failure();
    // }
    // auto type = module::getStorageType(FaOp.getOutput());
    // int64_t batch = FaOp.getBatch();
    // int64_t head = FaOp.getHead();
    // int64_t mq = FaOp.getMq();
    // int64_t mk = FaOp.getMk();
    // // add buffer
    // std::vector<int64_t> buffer_shape = {batch, mq, head, mk};
    // auto buffer_type = RankedTensorType::get(buffer_shape, type);
    // auto buffer = tpu::BufferOp::create(FaOp, buffer_type);
    // FaOp.getBufferMutable().assign(buffer);
    return success();
  }
};

class GatherElementsGlobalBuffer
    : public OpRewritePattern<tpu::GatherElementsOp> {
public:
  using OpRewritePattern<tpu::GatherElementsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::GatherElementsOp GatherElementsOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(GatherElementsOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() && !module::isBM1690Family()) {
      return failure();
    }
    auto buffer_type =
        GatherElementsOp.getIndices().getType().cast<RankedTensorType>();
    auto buffer = tpu::BufferOp::create(GatherElementsOp, buffer_type);
    GatherElementsOp.setOperand(3, buffer);
    return success();
  }
};

class LSTMGlobalBuffer : public OpRewritePattern<tpu::LSTMOp> {
public:
  using OpRewritePattern<tpu::LSTMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::LSTMOp lstmOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(lstmOp.getBuffer())) {
      return failure();
    }
    auto attr = lstmOp.parseParam();
    auto type = module::getStorageType(lstmOp.getInput());
    // add buffer
    std::vector<int64_t> buffer_shape = {5, attr.batch_size, attr.hidden_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(lstmOp, buffer_type);
    lstmOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == lstmOp.getBuffer() &&
             operand.getOwner() == lstmOp;
    });
    return success();
  }
};

class ReduceGlobalBuffer : public OpRewritePattern<tpu::ReduceOp> {
public:
  using OpRewritePattern<tpu::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(reduceOp.getBuffer())) {
      return failure();
    }
    auto attr = reduceOp.parseParam();
    if (attr.simplified == false) {
      UNREACHABLE_OP("Not Implemented", reduceOp);
    }
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      auto type = module::getStorageType(reduceOp.getInput());
      auto input_spec = BM168x::get_input_spec(reduceOp);
      auto output_spec = BM168x::get_output_spec(reduceOp);
      reduce_full_global_param_t param = {0};
      // need to check if it is dynamic mode
      auto run_mode = tpu::getRunMode(reduceOp.getOperation());
      if (run_mode == tpu::RunMode::TPU_STATIC) {
        BM168x::fix_shape(input_spec->at(0), {attr.outer_n, attr.outer_c,
                                              attr.axis_dims, attr.inner_dims});
        BM168x::fix_shape(output_spec->at(0),
                          {attr.outer_n, attr.outer_c, 1, attr.inner_dims});
        param.spec.common.axis_num = 1;
        param.spec.common.axis[0] = 2;
      } else if (run_mode == tpu::RunMode::TPU_DYNAMIC) {
        auto &&axes = reduceOp.getAxes();
        param.spec.common.axis_num = axes.size();
        for (int i = 0; i < axes.size(); i++) {
          param.spec.common.axis[i] = (axes[i].cast<IntegerAttr>().getInt());
        }
        param.spec.common.keep_dims = reduceOp.getKeepdims() ? 1 : 0;
      }

      param.spec.common.method = BM168x::get_reduce_type(reduceOp.getMode());
      param.if_getting_buffer_size = true;
      uint64_t buffer_size = 0;
      param.buffer_size_ptr = &buffer_size;
      BM168x::call_global_func("backend_api_reduce_full_global", &param,
                               sizeof(param), input_spec->data(),
                               output_spec->data());
      if (buffer_size > 0) {
        std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
        auto buffer_type = RankedTensorType::get(buffer_shape, type);
        auto buffer = tpu::BufferOp::create(reduceOp, buffer_type);
        reduceOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
          return operand.get() == reduceOp.getBuffer() &&
                 operand.getOwner() == reduceOp;
        });
      }
    } else if (module::isBM1684Family()) {
      auto type = module::getStorageType(reduceOp.getInput());
      auto in_addr = module::getAddress(reduceOp.getInput());
      auto out_addr = module::getAddress(reduceOp.getOutput());
      int input_dims = module::getShape(reduceOp.getInput()).size();
      uint64_t buffer_size = 0;
      uint64_t *buffer_size_ptr = &buffer_size;
      uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];
      for (auto v : llvm::enumerate(module::getShape(reduceOp.getInput())))
        input_shape[v.index()] = (uint32_t)v.value();
      auto &&axes = reduceOp.getAxes();
      int axis_num = axes.size();
      int axis_list[axis_num];
      for (int i = 0; i < axes.size(); i++)
        axis_list[i] = (axes[i].cast<IntegerAttr>().getInt());
      int method = BM1684::get_reduce_type(reduceOp.getMode());
      uint64_t buffer_addr = module::getAddress(reduceOp.getBuffer());
      if (BM1684::getDataType(reduceOp.getInput()) == DTYPE_FP32 ||
          BM1684::getDataType(reduceOp.getInput()) == DTYPE_INT32 ||
          BM1684::getDataType(reduceOp.getInput()) == DTYPE_UINT32) {
        BM1684::instance().dl_nodechip_reduce_full_v3(
            in_addr, out_addr, (const uint32_t *)input_shape, input_dims,
            axis_list, axis_num, method, buffer_addr, buffer_size_ptr,
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
      } else {
        int keep_dims = reduceOp.getKeepdims() ? 1 : 0;
        buffer_size =
            BM1684::instance().dl_nodechip_reduce_get_buffer_size_fix8b(
                input_shape, input_dims, axis_list, axis_num, method,
                keep_dims);
      }
      if (buffer_size > 0) {
        std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
        auto buffer_type = RankedTensorType::get(buffer_shape, type);
        auto buffer = tpu::BufferOp::create(reduceOp, buffer_type);
        reduceOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
          return operand.get() == reduceOp.getBuffer() &&
                 operand.getOwner() == reduceOp;
        });
      }
    }

    return success();
  }
};

class SliceGlobalBuffer : public OpRewritePattern<tpu::SliceOp> {
public:
  using OpRewritePattern<tpu::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(sliceOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }
    if (!module::isUniformQuantized(sliceOp.getInput())) {
      return failure();
    }
    if (!module::isUniformQuantized(sliceOp.getOutput())) {
      return failure();
    }
    auto p = sliceOp.parseParam();
    uint64_t buffer_size = 0;
    int shape_dim = p.is_4.size();
    // melloc
    int *input_shape = new int[MAX_SHAPE_DIMS];
    int *begin_index = new int[MAX_SHAPE_DIMS];
    int *end_index = new int[MAX_SHAPE_DIMS];
    int *stride = new int[MAX_SHAPE_DIMS];
    // assign param and call func to get buffer size
    for (int i = 0; i < shape_dim; ++i) {
      input_shape[i] = p.is_4[i];
      begin_index[i] = p.offset_4[i];
      end_index[i] = p.os_4[i] * p.step_4[i] + p.offset_4[i];
      stride[i] = p.step_4[i];
    }
    BM1684::instance().dl_nodechip_stride_slice_fix8b(
        0, 0, 0, 0, &buffer_size, input_shape, shape_dim, STORE_MODE_4N,
        STORE_MODE_4N, 0, 0, begin_index, end_index, stride, 0, NULL);
    // release
    delete[] input_shape;
    delete[] begin_index;
    delete[] end_index;
    delete[] stride;
    if (!buffer_size)
      return failure();
    // create bufferOp
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto type = module::getStorageType(sliceOp.getOutput());
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(sliceOp, buffer_type);
    sliceOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == sliceOp.getBuffer() &&
             operand.getOwner() == sliceOp;
    });
    return success();
  }
};

class ReshapeGlobalBuffer : public OpRewritePattern<tpu::ReshapeOp> {
public:
  using OpRewritePattern<tpu::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // only used for 4N ndim reshape!
    if (!module::isBM1684Family()) {
      return failure();
    }
    if (!module::isNone(reshapeOp.getBuffer())) {
      return failure();
    }
    if (!module::isUniformQuantized(reshapeOp.getInput())) {
      return failure();
    }
    if (!module::isUniformQuantized(reshapeOp.getOutput())) {
      return failure();
    }
    int64_t in, ic, ih, iw, on, oc, oh, ow;
    module::getNCHW(reshapeOp.getInput(), in, ic, ih, iw);
    module::getNCHW(reshapeOp.getOutput(), on, oc, oh, ow);
    if (on == in) {
      return failure();
    }

    uint64_t buffer_size = on * oc * oh * ow;
    ;

    // create bufferOp
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto type = module::getStorageType(reshapeOp.getOutput());
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(reshapeOp, buffer_type);
    reshapeOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == reshapeOp.getBuffer() &&
             operand.getOwner() == reshapeOp;
    });
    return success();
  }
};
class SoftmaxGlobalBuffer : public OpRewritePattern<tpu::SoftmaxOp> {
public:
  using OpRewritePattern<tpu::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::SoftmaxOp softmaxOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(softmaxOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }
    if (!module::isUniformQuantized(softmaxOp.getInput())) {
      return failure();
    }
    int64_t n, c, h, w;
    module::getNCHW(softmaxOp.getInput(), n, c, h, w);
    std::vector<int64_t> buffer_shape = {ceiling_func(n, (int64_t)4), c, h, w};
    auto type = module::getStorageType(softmaxOp.getOutput());
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(softmaxOp, buffer_type);
    softmaxOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == softmaxOp.getBuffer() &&
             operand.getOwner() == softmaxOp;
    });
    return success();
  }
};

class PermuteGlobalBuffer : public OpRewritePattern<tpu::PermuteOp> {
public:
  using OpRewritePattern<tpu::PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PermuteOp permuteOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(permuteOp.getBuffer())) {
      return failure();
    }
    uint64_t buffer_size = 0;
    auto attr = permuteOp.parseParam();
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      transpose_param_t param = {0};
      param.if_getting_buffer_size = 1;
      for (int i = 0, n = attr.order_fix.size(); i < n; ++i) {
        param.spec.order[i] = attr.order_fix[i];
      }
      param.buffer_size_ptr = &buffer_size;
      auto input_spec = BM168x::get_input_spec(permuteOp);
      auto output_spec = BM168x::get_output_spec(permuteOp);
      BM168x::fix_shape(input_spec->at(0), attr.in_shape_fix);
      BM168x::fix_shape(output_spec->at(0), attr.out_shape_fix);
      BM168x::call_global_func("backend_api_transpose_global", &param,
                               sizeof(param), input_spec->data(),
                               output_spec->data());
    } else if (module::isBM1684Family()) {
      // melloc
      uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];
      int *order = new int[MAX_SHAPE_DIMS];
      uint64_t *buffer_size_ptr = &buffer_size;

      auto input = permuteOp.getInput();
      auto output = permuteOp.getOutput();
      i32_array_t in_order = module::getI32Array(permuteOp.getOrder());
      auto input_addr = module::getAddress(input);
      auto output_addr = module::getAddress(output);
      int input_dims = module::getShape(input).size();
      auto input_dtype = BM1684::getDataType(input);
      auto output_dtype = BM1684::getDataType(output);
      int type_len = BM1684::getFmtBytes(input_dtype);
      int store_mode;
      for (auto v : llvm::enumerate(module::getShape(input)))
        input_shape[v.index()] = (uint32_t)v.value();
      memcpy(order, in_order->data(), (*in_order).size() * sizeof(int));
      if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32 ||
          input_dtype == DTYPE_UINT32) {
        store_mode = STORE_MODE_1N;
        BM1684::instance().dl_nodechip_transpose(
            input_addr, output_addr, input_shape, order, input_dims, type_len,
            store_mode, 0, buffer_size_ptr,
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
      } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
        store_mode = STORE_MODE_4N;
        assert(output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8);
        BM1684::instance().dl_nodechip_transpose_fix8b(
            input_addr, output_addr, input_shape, order, input_dims, store_mode,
            store_mode, 0, buffer_size_ptr,
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
      } else {
        UNREACHABLE_OP("Not Implemented", permuteOp);
        return failure();
      }
      // release
      delete[] input_shape;
      delete[] order;
      buffer_size_ptr = NULL;
    } else {
      return failure();
    }
    auto type = module::getStorageType(permuteOp.getInput());
    // add buffer
    if (buffer_size > 0) {
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(permuteOp, buffer_type);
      permuteOp.setOperand(1, buffer);
      return success();
    }
    return failure();
  }
};

class NonZeroGlobalBuffer : public OpRewritePattern<tpu::NonZeroOp> {
public:
  using OpRewritePattern<tpu::NonZeroOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::NonZeroOp op,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(op.getBuffer())) {
      return failure();
    }
    auto type = module::getStorageType(op.getInput());
    // add buffer
    auto buffer_type =
        RankedTensorType::get(module::getShape(op.getInput()), type);
    auto buffer = tpu::BufferOp::create(op, buffer_type);
    op.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == op.getBuffer() && operand.getOwner() == op;
    });
    return success();
  }
};

class InterpGlobalBuffer : public OpRewritePattern<tpu::InterpOp> {
public:
  using OpRewritePattern<tpu::InterpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::InterpOp interpOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily() && !module::isBM1690Family())
      return failure();

    if (!module::isNone(interpOp.getBuffer())) {
      return failure();
    }

    interp_global_param_t param = {0};
    param.if_getting_buffer_size = true;
    uint64_t buffer_size = 0;
    param.buffer_size_ptr = &buffer_size;
    auto input_spec = BM168x::get_input_spec(interpOp);
    auto output_spec = BM168x::get_output_spec(interpOp);
    BM168x::call_global_func("backend_api_interp_global", &param, sizeof(param),
                             input_spec->data(), output_spec->data());
    // add buffer
    if (buffer_size > 0) {
      auto type = ::mlir::Builder(getContext()).getIntegerType(8);
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(interpOp, buffer_type);
      interpOp.setOperand(2, buffer);
      return success();
    }

    return failure();
  }
};

class MatMulGlobalBuffer : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern<tpu::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::MatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isBM1690Family())
      return failure();

    if (!module::isNone(matmulOp.getBuffer()))
      return failure();

    if(module::getCoreNum() == 1 || !matmulOp.supports_multi_core() || !matmulOp.getMultiCore())
      return failure();


    auto p = matmulOp.parseParam();
    fc_global_spec_t spec = {0};
    memset(&spec, 0, sizeof(spec));
    spec.if_getting_buffer_size = true;
    uint64_t buffer_size = 0;
    spec.buffer_size_ptr = &buffer_size;
    spec.if_relu = p.do_relu;
    spec.relu_limit = p.relu_limit;
    spec.have_bias = p.with_bias;
    spec.requant_mode = -1;
    spec.R_transpose = p.right_transpose;
    if (module::isUniformQuantized(matmulOp.getInput())) {
      spec.rshift = 0;
      spec.is_asymmetric = 1;
      spec.rzp_is_const = 1;
      spec.rzp_const_val = p.right_zp;
      spec.izp_const_val = p.input_zp;
      if (module::isUniformQuantized(matmulOp.getOutput())) {
        auto rshift_v = module::getI64Array(matmulOp.getRshifts(), 1, 0);
        auto multiplier_v = module::getI64Array(matmulOp.getMultipliers(), 1, 1);
        assert(rshift_v->size() == 1);
        assert(multiplier_v->size() == 1);
        spec.requant_mode = static_cast<int>(matmulOp.getQuantMode());
        spec.mul_val = multiplier_v->at(0);
        spec.shift_val = -rshift_v->at(0);
        auto output_type = module::getUniformQuantizedType(matmulOp.getOutput());
        spec.offset_val = output_type.getZeroPoint();
        spec.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
      }
    }
    auto input_spec = BM168x::get_input_spec(matmulOp);
    auto output_spec = BM168x::get_output_spec(matmulOp);
    BM168x::call_global_func("backend_api_fc_multi_core_global", &spec, sizeof(spec),
                             input_spec->data(), output_spec->data());
    if (buffer_size > 0) {
      auto type = ::mlir::Builder(getContext()).getIntegerType(8);
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(matmulOp, buffer_type, tpu::BufferType::L2);
      matmulOp.setOperand(module::isNone(matmulOp.getBias()) ? 2 : 3, buffer);
      return success();
    }
    return failure();
  }
};

class GridSamplerBuffer : public OpRewritePattern<tpu::GridSamplerOp> {
public:
  using OpRewritePattern<tpu::GridSamplerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::GridSamplerOp gridSamplerOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    if (!module::isNone(gridSamplerOp.getBuffer())) {
      return failure();
    }
    auto type = ::mlir::Builder(getContext()).getIntegerType(8);
    auto output_shape = module::getShape(gridSamplerOp.getOutput());
    auto coeff_dtype_size = module::getDtypeSize(gridSamplerOp.getOutput());
    auto dim = output_shape.size();
    int f32_size = 4;
    int64_t buffer_size;
    if (dim == 4) {
      buffer_size = output_shape[0] * align_up(output_shape[2] * output_shape[3], Arch::NPU_NUM) * (2 * f32_size + 4 * coeff_dtype_size);
    } else if (dim == 5) {
      buffer_size = output_shape[0] * align_up(output_shape[2] * output_shape[3] * output_shape[4], Arch::NPU_NUM) * (3 * f32_size + 8 * coeff_dtype_size);
    } else {
      return failure();
    }
    auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
    auto buffer = tpu::BufferOp::create(gridSamplerOp, buffer_type);
    gridSamplerOp.setOperand(gridSamplerOp.getNumOperands() - 1, buffer);
    return success();
  }
};

class DeformGatherGlobalBuffer : public OpRewritePattern<tpu::DeformGatherOp> {
public:
  using OpRewritePattern<tpu::DeformGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::DeformGatherOp Op,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(Op.getBuffer())) {
      return failure();
    }
    if (!(module::isBM1684XFamily() || module::isBM1690Family())) {
      return failure();
    }
    deform_gather_attr_t p = Op.parseParam();
    uint64_t buffer_size = 0;
    auto conved_H =
        ((p.ih - (p.dh * (p.kh - 1) + 1) + p.pht + p.phb) / p.sh + 1);
    auto conved_W =
        ((p.iw - (p.dw * (p.kw - 1) + 1) + p.pwl + p.pwr) / p.sw + 1);
    auto full_size = p.kh * p.kw * conved_H * conved_W * sizeof(float);
    buffer_size = 2 * p.n * p.deform_groups * full_size;
    if (p.use_mask)
      buffer_size = std::max(buffer_size, (uint64_t)p.n * p.ic *
                                              p.deform_groups * full_size);
    auto type = module::getStorageType(Op.getInput());
    auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
    auto buffer = tpu::BufferOp::create(Op, buffer_type);
    Op.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == Op.getBuffer() && operand.getOwner() == Op;
    });
    return success();
  }
};
class Space2BatchGlobalBuffer : public OpRewritePattern<tpu::Space2BatchOp> {
public:
  using OpRewritePattern<tpu::Space2BatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::Space2BatchOp space2batchOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(space2batchOp.getBuffer())) {
      return failure();
    }
    uint64_t buffer_size = 0;
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      llvm_unreachable("Not supported now");
      return failure();
    } else if (module::isBM1684Family()) {
      // melloc
      uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];

      auto input = space2batchOp.getInput();
      auto output = space2batchOp.getOutput();
      auto pads_v = module::getI64Array(space2batchOp.getPads());
      int64_t pad_top = pads_v->at(0);
      int64_t pad_bottom = pads_v->at(1);
      int64_t pad_left = pads_v->at(2);
      int64_t pad_right = pads_v->at(3);

      int input_dims = module::getShape(input).size();
      auto input_dtype = BM1684::getDataType(input);
      auto output_dtype = BM1684::getDataType(output);
      int type_len = BM1684::getFmtBytes(input_dtype);
      int store_mode;
      for (auto v : llvm::enumerate(module::getShape(input)))
        input_shape[v.index()] = (uint32_t)v.value();
      if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32 ||
          input_dtype == DTYPE_UINT32) {
        int height_pad = input_shape[2] + pad_top + pad_bottom;
        int width_pad = input_shape[3] + pad_left + pad_right;
        buffer_size =
            input_shape[0] * input_shape[1] * height_pad * width_pad * type_len;
      } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
        store_mode = STORE_MODE_4N;
        assert(output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8);
        int block_h = (int)space2batchOp.getBlockH();
        int block_w = (int)space2batchOp.getBlockW();
        int block_size[2] = {block_h, block_w};
        int pad_sizes[4] = {(int)pad_top, (int)pad_bottom, (int)pad_left,
                            (int)pad_right};
        BM1684::instance().dl_nodechip_space2batch_fix8b(
            0, 0, 0, &buffer_size, (int *)input_shape, input_dims, store_mode,
            store_mode, block_size, pad_sizes, NULL, NULL);
      } else {
        UNREACHABLE_OP("Not Implemented", space2batchOp);
        return failure();
      }
      // release
      delete[] input_shape;
    } else {
      UNREACHABLE_OP("Not Implemented", space2batchOp);
      return failure();
    }
    auto type = module::getStorageType(space2batchOp.getInput());
    // add buffer
    if (buffer_size > 0) {
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(space2batchOp, buffer_type);
      space2batchOp.setOperand(1, buffer);
      return success();
    }
    return failure();
  }
};

class Batch2SpaceGlobalBuffer : public OpRewritePattern<tpu::Batch2SpaceOp> {
public:
  using OpRewritePattern<tpu::Batch2SpaceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::Batch2SpaceOp batch2spaceOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(batch2spaceOp.getBuffer())) {
      return failure();
    }
    uint64_t buffer_size = 0;
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      llvm_unreachable("Not supported now");
      return failure();
    } else if (module::isBM1684Family()) {
      // melloc
      uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];

      auto input = batch2spaceOp.getInput();
      auto output = batch2spaceOp.getOutput();
      auto crops_v = module::getI64Array(batch2spaceOp.getCrops());
      int64_t crop_top = crops_v->at(0);
      int64_t crop_bottom = crops_v->at(1);
      int64_t crop_left = crops_v->at(2);
      int64_t crop_right = crops_v->at(3);

      int input_dims = module::getShape(input).size();
      auto input_dtype = BM1684::getDataType(input);
      auto output_dtype = BM1684::getDataType(output);
      int type_len = BM1684::getFmtBytes(input_dtype);
      int store_mode;
      for (auto v : llvm::enumerate(module::getShape(input)))
        input_shape[v.index()] = (uint32_t)v.value();
      if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32 ||
          input_dtype == DTYPE_UINT32) {
        buffer_size = input_shape[0] * input_shape[1] * input_shape[2] *
                      input_shape[3] * type_len;
      } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
        store_mode = STORE_MODE_4N;
        assert(output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8);
        int block_h = (int)batch2spaceOp.getBlockH();
        int block_w = (int)batch2spaceOp.getBlockW();
        int block_size[2] = {block_h, block_w};
        int pad_sizes[4] = {(int)crop_top, (int)crop_bottom, (int)crop_left,
                            (int)crop_right};
        BM1684::instance().dl_nodechip_batch2space_fix8b(
            0, 0, 0, 0, &buffer_size, (int *)input_shape, input_dims,
            store_mode, store_mode, block_size, pad_sizes, NULL, NULL);
      } else {
        UNREACHABLE_OP("Not Implemented", batch2spaceOp);
        return failure();
      }
      // release
      delete[] input_shape;
    } else {
      UNREACHABLE_OP("Not Implemented", batch2spaceOp);
      return failure();
    }
    auto type = module::getStorageType(batch2spaceOp.getInput());
    // add buffer
    if (buffer_size > 0) {
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(batch2spaceOp, buffer_type);
      batch2spaceOp.setOperand(1, buffer);
      return success();
    }
    return failure();
  }
};

class TileGlobalBuffer : public OpRewritePattern<tpu::TileOp> {
public:
  using OpRewritePattern<tpu::TileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::TileOp TileOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(TileOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family() && (!module::isBM1684XFamily()) &&
        (!module::isBM1690Family())) {
      UNREACHABLE_OP("Not Implemented", TileOp);
      return failure();
    }
    auto input_shape = module::getShape(TileOp.getInput());
    int input_dim = input_shape.size();
    auto output_shape = module::getShape(TileOp.getOutput());
    int tile_coeff[8];
    auto type = module::getStorageType(TileOp.getInput());
    int64_t type_len = module::getDtypeSize(TileOp.getInput());
    uint64_t buffer_size = 0;
    int tile_count = 0;
    int min_tile = output_shape[0] / input_shape[0];
    uint64_t total_size = 1;
    for (int i = 0; i < input_dim; ++i) {
      tile_coeff[i] =
          output_shape[i] < 0 ? 1 : output_shape[i] / input_shape[i];
      if (tile_coeff[i] > 1)
        tile_count++;
      if (tile_coeff[i] < min_tile)
        min_tile = tile_coeff[i];
      total_size *= output_shape[i];
    }
    if (type_len > 0) {
      if (tile_count > 1) {
        buffer_size = total_size / min_tile * type_len;
      }
      if (buffer_size > 0) {
        std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
        auto buffer_type = RankedTensorType::get(buffer_shape, type);
        auto buffer = tpu::BufferOp::create(TileOp, buffer_type);
        TileOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
          return operand.get() == TileOp.getBuffer() &&
                 operand.getOwner() == TileOp;
        });
        return success();
      }
    }
    return failure();
  }
};

class IndexPutGlobalBuffer : public OpRewritePattern<tpu::IndexPutOp> {
public:
  using OpRewritePattern<tpu::IndexPutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::IndexPutOp IndexPutOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(IndexPutOp.getBuffer()) ||
        !IndexPutOp.getAccumulate()) {
      return failure();
    }
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    auto elment_num = module::getNumElements(IndexPutOp.getValues());
    auto type = module::getStorageType(IndexPutOp.getValues());
    // add buffer
    std::vector<int64_t> buffer_shape = {elment_num}; // double buffer
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(IndexPutOp, buffer_type);
    IndexPutOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == IndexPutOp.getBuffer() &&
             operand.getOwner() == IndexPutOp;
    });
    return success();
  }
};

class PadGlobalBuffer : public OpRewritePattern<tpu::PadOp> {
public:
  using OpRewritePattern<tpu::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(padOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }
    std::vector<int64_t> shape = module::getShape(padOp.getInput());
    if (shape.size() < 5 && !module::isUniformQuantized(padOp.getInput())) {
      return failure();
    }
    if (shape.size() <= 5) {
      std::vector<int64_t> buffer_shape;
      buffer_shape.push_back(ceiling_func(shape[0], (int64_t)4));
      for (int i = 1; i < shape.size(); ++i) {
        buffer_shape.push_back(shape[i]);
      }
      auto type = module::getStorageType(padOp.getOutput());
      auto buffer_type = RankedTensorType::get(buffer_shape, type);
      auto buffer = tpu::BufferOp::create(padOp, buffer_type);
      padOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
        return operand.get() == padOp.getBuffer() &&
               operand.getOwner() == padOp;
      });
      return success();
    } else {
      return failure();
    }
  }
};

class GatherGlobalBuffer : public OpRewritePattern<tpu::GatherOp> {
public:
  using OpRewritePattern<tpu::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::GatherOp GatherOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(GatherOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }
    auto elment_num = module::getNumElements(GatherOp.getInput());
    auto type = module::getStorageType(GatherOp.getInput());
    // add buffer
    std::vector<int64_t> buffer_shape = {2, elment_num}; // double buffer
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(GatherOp, buffer_type);
    GatherOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == GatherOp.getBuffer() &&
             operand.getOwner() == GatherOp;
    });
    return success();
  }
};

class Pool3DGlobalBuffer : public OpRewritePattern<tpu::Pool3DOp> {
public:
  using OpRewritePattern<tpu::Pool3DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::Pool3DOp Pool3DOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(Pool3DOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }

    auto elment_num = module::getNumElements(Pool3DOp.getInput());
    auto type = module::getStorageType(Pool3DOp.getInput());
    // add buffer
    std::vector<int64_t> buffer_shape = {2, elment_num}; // double buffer
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(Pool3DOp, buffer_type);
    Pool3DOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == Pool3DOp.getBuffer() &&
             operand.getOwner() == Pool3DOp;
    });
    return success();
  }
};

class ScatterElementsGlobalBuffer
    : public OpRewritePattern<tpu::ScatterElementsOp> {
public:
  using OpRewritePattern<tpu::ScatterElementsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ScatterElementsOp ScatterElementsOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(ScatterElementsOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() && !module::isBM1690Family()) {
      return failure();
    }
    auto buffer_type =
        ScatterElementsOp.getIndices().getType().cast<RankedTensorType>();
    auto buffer = tpu::BufferOp::create(ScatterElementsOp, buffer_type);
    ScatterElementsOp.setOperand(4, buffer);
    return success();
  }
};

class ScatterNDGlobalBuffer : public OpRewritePattern<tpu::ScatterNDOp> {
public:
  using OpRewritePattern<tpu::ScatterNDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ScatterNDOp ScatterNDOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(ScatterNDOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() && !module::isBM1690Family()) {
      return failure();
    }
    auto buffer_type =
        // ScatterNDOp.getInputData().getType().cast<RankedTensorType>();
        RankedTensorType::get(module::getShape(ScatterNDOp.getInputData()),
                              rewriter.getI32Type());
    auto buffer = tpu::BufferOp::create(ScatterNDOp, buffer_type);
    ScatterNDOp.setOperand(3, buffer);
    return success();
  }
};

class NmsGlobalBuffer : public OpRewritePattern<tpu::NmsOp> {
public:
  using OpRewritePattern<tpu::NmsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::NmsOp NmsOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(NmsOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() ||
        (module::isBM1684XFamily() && !module::isBM1688())) {
      return failure();
    }

    int64_t buffer_size = BUFFER_SIZE;
    auto type = module::getStorageType(NmsOp.getInputs()[0]);
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(NmsOp, buffer_type);
    NmsOp.setOperand(NmsOp.getNumOperands() - 1, buffer);
    return success();
  }
};

class YoloDetectionGlobalBuffer
    : public OpRewritePattern<tpu::YoloDetectionOp> {
public:
  using OpRewritePattern<tpu::YoloDetectionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::YoloDetectionOp yoloDetectionOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(yoloDetectionOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() ||
        (module::isBM1684XFamily() && !module::isBM1688())) {
      return failure();
    }
    auto process = module::getPostprocess();
    if (process.starts_with("yolov5")) {
      return failure();
    }
    int64_t buffer_size = BUFFER_SIZE;
    auto type = module::getStorageType(yoloDetectionOp.getInputs()[0]);
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(yoloDetectionOp, buffer_type);
    yoloDetectionOp.setOperand(yoloDetectionOp.getNumOperands() - 1, buffer);
    return success();
  }
};

class DetectionOutputGlobalBuffer
    : public OpRewritePattern<tpu::DetectionOutputOp> {
public:
  using OpRewritePattern<tpu::DetectionOutputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::DetectionOutputOp detectionOutputOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(detectionOutputOp.getBuffer())) {
      return failure();
    }

    if (!module::isBM1684XFamily() ||
        (module::isBM1684XFamily() && !module::isBM1688())) {
      return failure();
    }

    int64_t buffer_size = BUFFER_SIZE;
    auto type = module::getStorageType(detectionOutputOp.getInputs()[0]);
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(detectionOutputOp, buffer_type);
    detectionOutputOp.setOperand(detectionOutputOp.getNumOperands() - 1,
                                 buffer);
    return success();
  }
};

class TopKGlobalBuffer : public OpRewritePattern<tpu::TopKOp> {
public:
  using OpRewritePattern<tpu::TopKOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::TopKOp TopKOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(TopKOp.getBufferVal()) ||
        !module::isNone(TopKOp.getBufferIdx())) {
      return failure();
    }

    if (!module::isBM1684XFamily()) {
      return failure();
    }

    int64_t num_elements = module::getNumElements(TopKOp.getInput());
    auto val_type = module::getStorageType(TopKOp.getValues());
    auto idx_type = module::getStorageType(TopKOp.getIndices());
    std::vector<int64_t> buffer_shape = {num_elements};
    auto val_buffer_type = RankedTensorType::get(buffer_shape, val_type);
    auto idx_buffer_type = RankedTensorType::get(buffer_shape, idx_type);

    OpBuilder builder(TopKOp->getContext());
    builder.setInsertionPoint(TopKOp);
    auto val_loc = module::getLocLike(TopKOp, "buffer_val");
    auto val_buf_op = builder.create<tpu::BufferOp>(val_loc, val_buffer_type);
    auto idx_loc = module::getLocLike(TopKOp, "buffer_idx");
    auto idx_buf_op = builder.create<tpu::BufferOp>(idx_loc, idx_buffer_type);

    TopKOp.setOperand(TopKOp.getNumOperands() - 2, val_buf_op);
    TopKOp.setOperand(TopKOp.getNumOperands() - 1, idx_buf_op);
    return success();
  }
};

class SortGlobalBuffer : public OpRewritePattern<tpu::SortOp> {
public:
  using OpRewritePattern<tpu::SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::SortOp op,
                                PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    auto _op = op.getOperation();
    auto input_spec = BM168x::get_input_spec(_op);
    auto output_spec = BM168x::get_output_spec(_op);
    sort_per_dim_param_t param = {0};
    param.buffer_addr = module::getAddress(op.getBuffer());
    param.axis = op.getAxis();
    param.descending = op.getDescending();
    param.is_argsort = module::isNone(op.getValues());
    int64_t buffer_size = BM168x::call_global_bfsz_func(
      "backend_api_sort_per_dim_global_bfsz",
      &param, sizeof(param),
      input_spec->data(), output_spec->data());
    if (buffer_size) {
      std::vector<int64_t> buffer_shape = {buffer_size};
      auto buffer_type = RankedTensorType::get(buffer_shape, rewriter.getI8Type());
      OpBuilder builder(op->getContext());
      builder.setInsertionPoint(op);
      auto loc = module::getLocLike(op, "buffer");
      auto buf_op = builder.create<tpu::BufferOp>(loc, buffer_type);
      op.setOperand(op.getNumOperands() - 1, buf_op);
      return success();
    } else {
      return failure();
    }
  }
};

#include "tpu_mlir/Support/CustomLayer.h"
class CustomGlobalBuffer : public OpRewritePattern<tpu::CustomOp> {
public:
  using OpRewritePattern<tpu::CustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::CustomOp customOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    auto op = customOp.getOperation();
    auto input_spec = BM168x::get_input_spec(op);
    auto output_spec = BM168x::get_output_spec(op);
    // call backend api according to the custom op name
    std::string op_name = customOp.getName().str();
    std::string api_name = "backend_api_" + op_name + "_global_bfsz";
    // parse param of the custom op
    auto params = customOp.getParams();
    std::vector<custom_param_t> values;
    values.push_back({0});
    customOpProcessParam(params, values);
    int64_t buffer_size = BM168x::call_global_bfsz_custom_func(
                            api_name.c_str(), values.data(), values.size() * sizeof(custom_param_t),
                            input_spec->data(), output_spec->data());

    if (buffer_size > 0) {
      std::vector<int64_t> buffer_shape = {buffer_size};
      auto buffer_type = RankedTensorType::get(buffer_shape, rewriter.getI8Type());
      OpBuilder builder(customOp->getContext());
      builder.setInsertionPoint(customOp);
      auto loc = module::getLocLike(customOp, "buffer");
      auto buf_op = builder.create<tpu::BufferOp>(loc, buffer_type);
      customOp.setOperand(customOp.getNumOperands() - 1, buf_op);
      return success();
    } else {
      return failure();
    }
  }
};

class WhereGlobalBuffer : public OpRewritePattern<tpu::WhereOp> {
public:
  using OpRewritePattern<tpu::WhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::WhereOp WhereOp,
                                PatternRewriter &rewriter) const override {
    if (tpu::getRunMode(WhereOp) != tpu::RunMode::TPU_DYNAMIC) {
      return failure();
    }
    if (!module::isNone(WhereOp.getBuffer())) {
      return failure();
    }
    auto out_shape = module::getShape(WhereOp.getOutput());
    auto cond_shape = module::getShape(WhereOp.getCond());
    auto out_dim = out_shape.size();
    auto cond_reshape = shape_expand_dim(cond_shape, out_dim);
    int buffer_num = 0;
    for (int i = 0; i < out_dim; ++i) {
      if (out_shape[i] != cond_reshape[i]) {
        buffer_num ++;
        break;
      }
    }
    if (!WhereOp.getXIsConst()) {
      auto tbrn = WhereOp.getTbrn();
      auto tbrn_shape = module::getShape(tbrn);
      auto tbrn_reshape = shape_expand_dim(tbrn_shape, out_dim);
      for (int i = 0; i < out_dim; ++i) {
        if (out_shape[i] != tbrn_reshape[i]) {
          buffer_num ++;
          break;
        }
      }
    }
    if (!WhereOp.getYIsConst()) {
      auto fbrn = WhereOp.getFbrn();
      auto fbrn_shape = module::getShape(fbrn);
      auto fbrn_reshape = shape_expand_dim(fbrn_shape, out_dim);
      for (int i = 0; i < out_dim; ++i) {
        if (out_shape[i] != fbrn_reshape[i]) {
          buffer_num ++;
          break;
        }
      }
    }
    if (!buffer_num) {
      return failure();
    }
    auto elment_num = module::getNumElements(WhereOp.getOutput());
    auto type = module::getStorageType(WhereOp.getCond());
    // add buffer
    std::vector<int64_t> buffer_shape = {elment_num * 2 * buffer_num}; // double buffer for tile
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(WhereOp, buffer_type);
    WhereOp.setOperand(WhereOp.getNumOperands() - 1, buffer);
    return success();
  }
};

} // namespace bm168x

namespace tpu {
using namespace bm168x;
void populateGlobalBufferBM168xPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      GatherGlobalBuffer,
      GatherElementsGlobalBuffer,
      GRUGlobalBuffer,
      LSTMGlobalBuffer,
      ReduceGlobalBuffer,
      SliceGlobalBuffer,
      ReshapeGlobalBuffer,
      SoftmaxGlobalBuffer,
      PermuteGlobalBuffer,
      InterpGlobalBuffer,
      GridSamplerBuffer,
      Pool3DGlobalBuffer,
      NonZeroGlobalBuffer,
      DeformGatherGlobalBuffer,
      TileGlobalBuffer,
      IndexPutGlobalBuffer,
      PadGlobalBuffer,
      ScatterElementsGlobalBuffer,
      Space2BatchGlobalBuffer,
      Batch2SpaceGlobalBuffer,
      ScatterNDGlobalBuffer,
      NmsGlobalBuffer,
      YoloDetectionGlobalBuffer,
      DetectionOutputGlobalBuffer,
      TopKGlobalBuffer,
      SortGlobalBuffer,
      CustomGlobalBuffer,
      WhereGlobalBuffer
  >(patterns->getContext());
  // clang-format on
}

} // namespace tpu
} // namespace tpu_mlir
