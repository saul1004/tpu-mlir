//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/BM168x/BackendInterfaces.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.h"
#include "progressbar.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include <llvm/Support/Debug.h>
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"

#define DEBUG_TYPE "layer-group"
using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {
#define MAX_GROUP_CLUSTER (50)

#define GROUP_CHECK_RETURN(val)                                                \
  {                                                                            \
    if (val) {                                                                 \
      llvm::errs() << "layer group is valid";                                  \
      return true;                                                             \
    } else {                                                                   \
      llvm::errs() << "layer group is invalid";                                \
      return false;                                                            \
    }                                                                          \
  }

bool opt_cost_all = false;

bool Sort_by_int(const std::pair<Value, int64_t> &v1,
                 const std::pair<Value, int64_t> &v2) {
  return v1.second < v2.second; // ��������
}

bool pair_op_int_Sort_by_int(const std::pair<Operation *, int> &v1,
                             const std::pair<Operation *, int> &v2) {
  return v1.second < v2.second;
}

// set GROUP_3D if there is 3DOp
static bool can_be_group_3d(std::vector<Operation *> &group_ops) {
  for (auto op : group_ops) {
    if (isa<Conv3DOp, Pool3DOp>(op)) {
      return true;
    }
    if (auto layernorm_op = dyn_cast<LayerNormOp>(op)) {
      if (module::getShape(layernorm_op.getInput()).size() == 5) {
        return true;
      }
    }
  }
  return false;
}

// set GROUP_NORMAL if not all ops should meet the conditions
// 1. op is eltwise-op or only the last dim cannot split
// 2. C is too small to fully utilize NPU and H is better
//    or N*C*H could be divided by NPU_NUM
static bool can_be_group_small_c(std::vector<Operation *> &group_ops) {
  auto ranmode = getRunMode(group_ops[0]);
  if (ranmode == RunMode::TPU_DYNAMIC) {
    return false;
  }
  for (auto op : group_ops) {
    if (!isa<ActiveOp, AddOp, CastOp, LayerNormOp, MulConstOp, MatMulOp,
             SoftmaxOp, RMSNormOp, ReshapeOp, LutOp>(op)) {
      return false;
    }
    auto shape = module::getShape(op->getOperand(0));
    if (auto op_ = dyn_cast<LayerNormOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
    } else if (isa<AddOp>(op)) {
      auto shapeB = module::getShape(op->getOperand(1));
      if (shape != shapeB) {
        return false;
      }
    } else if (auto op_ = dyn_cast<SoftmaxOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
    } else if (auto op_ = dyn_cast<MatMulOp>(op)) {
      auto hdim_is_batch = op_.getHdimIsBatch();
      if (hdim_is_batch) {
        return false;
      }
    } else if (auto op_ = dyn_cast<ReshapeOp>(op)) {
      auto ishape = module::getShape(op_.getInput());
      auto oshape = module::getShape(op_.getOutput());
      if (!(ishape.size() > 2 && oshape.size() > 2 && ishape[0] == oshape[0] &&
            ishape[1] == oshape[1])) {
        return false;
      }
    }

    if ((shape.size() == 4 &&
         shape[0] * shape[1] * shape[2] % Arch::NPU_NUM == 0) ||
        (shape.size() == 5 &&
         shape[0] * shape[1] * shape[2] * shape[3] % Arch::NPU_NUM == 0)) {
      continue;
    }
    if ((shape.size() == 3 && shape[0] > 1 && shape[1] == 197)) {
      continue;
    }

    if (!(((shape.size() == 5 && shape[3] > shape[1]) ||
           (shape.size() == 4 && shape[2] > shape[1])) &&
          shape[1] < Arch::NPU_NUM / 2)) {
      return false;
    }
  }
  return true;
}

static bool can_be_group_mm(std::vector<Operation *> &group_ops) {
  for (auto op : group_ops) {
    if (!isa<ActiveOp, AddOp, CastOp, LayerNormOp, MulConstOp, MatMulOp, MulOp,
             ReshapeOp, SoftmaxOp, AttentionOp, RMSNormOp, MulShiftOp>(op)) {
      return false;
    }
    auto shape = module::getShape(op->getOperand(0));
    if (auto op_ = dyn_cast<LayerNormOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
      //    } else if (isa<AddOp, MulOp>(op)) {
      //      auto shapeB = module::getShape(op->getOperand(1));
      //      if (shape != shapeB) {
      //        return false;
      //      }
    } else if (auto op_ = dyn_cast<ReshapeOp>(op)) {
      auto ishape = module::getShape(op_.getInput());
      auto oshape = module::getShape(op_.getOutput());
      if (!(ishape.size() > 2 && oshape.size() > 2 && ishape[0] == oshape[0] &&
            ishape[1] == oshape[1])) {
        return false;
      }
    } else if (auto op_ = dyn_cast<SoftmaxOp>(op)) {
      if (op_.getAxis() != shape.size() - 1) {
        return false;
      }
    } else if (auto op_ = dyn_cast<MatMulOp>(op)) {
      auto left_trans = op_.getLeftTranspose();
      auto right_trans = op_.getRightTranspose();
      if (left_trans && right_trans) {
        return false;
      }
    } else if (auto op_ = dyn_cast<AttentionOp>(op)) {
      if (module::isNone(op_.getKeys())) {
        return false;
      }
    }
  }

  return true;
}

static void set_group_type(LgInfo &lg_info) {
  lg_info.type = GROUP_NORMAL;
  if (lg_info.group_ops.size() == 1) {
    return;
  }

  if (can_be_group_3d(lg_info.group_ops)) {
    lg_info.type = GROUP_3D;
    return;
  }

  if (module::isCV18xx() || module::isBM1684Family()) {
    // cv18xx only support GROUP_NORMAL
    lg_info.type = GROUP_NORMAL;
    return;
  }

  if (can_be_group_small_c(lg_info.group_ops)) {
    lg_info.type = GROUP_SMALL_C;
    return;
  }

  if (can_be_group_mm(lg_info.group_ops)) {
    lg_info.type = GROUP_MM;
    return;
  }
}

static void get_layer_group(LgInfo &lg_info,
                            const std::vector<Operation *> &base_group,
                            int64_t left, int64_t right) {
  lg_info.clear();
  for (int idx = left; idx <= right; ++idx) {
    lg_info.group_ops.push_back(base_group[idx]);
  }
  lg_info.update_group_io(LgPass::OPTIONS.opt);
  set_group_type(lg_info);
}

GroupMethod::GroupMethod(int64_t opt) {
  if (module::isCV18xx()) {
    Cv18xxCycleCalculator *cyc_ptr = new Cv18xxCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  } else {
    Bm168xCycleCalculator *cyc_ptr = new Bm168xCycleCalculator();
    cycle_calculator_ = std::shared_ptr<CycleCalculator>(cyc_ptr);
  }
  MAX_COST = llvm::maxIntN(64);
  opt_ = opt;
}

int64_t GroupMethod::get_max_cluster_size(int64_t layer_num) {
  return std::max((int64_t)(layer_num / MAX_GROUP_CLUSTER), (int64_t)1);
}

int64_t GroupMethod::cost_add(int64_t cost0, int64_t cost1) {
  if (cost0 == MAX_COST || cost1 == MAX_COST) {
    return MAX_COST;
  } else {
    return (cost0 + cost1);
  }
}

bool GroupMethod::group_one_layer_proc(const LgInfo &lg_info, bool calc_cost,
                                       int64_t *group_cost) {
  if (lg_info.group_ops.size() == 1) {
    if (calc_cost) {
      *group_cost =
          cycle_calculator_->getGlobalLayerCycle(lg_info.group_ops.back());
    }
    return true;
  }
  return false;
}

bool is_binary_shape_value(Operation *op) {
  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MinOp,
          tpu::MaxOp>(op)) {
    auto l_shape = module::getShape(op->getOperand(0));
    auto r_shape = module::getShape(op->getOperand(1));
    if (l_shape.size() == 5 && l_shape[2] != r_shape[2])
      return true;
    else
      return false;
  } else {
    return false;
  }
}

void tmp_group_into_base(std::vector<std::vector<Operation *>> &base_groups,
                         std::vector<Operation *> &group, Operation *op,
                         bool &is_binary) {
  if (isa<Conv3DOp, Pool3DOp>(op) && is_binary) {
    std::vector<Operation *> tmp_group;
    for (auto tmp_op : group) {
      if (!is_binary_shape_value(tmp_op)) {
        tmp_group.push_back(tmp_op);
      } else {
        if (!tmp_group.empty()) {
          base_groups.push_back(tmp_group);
          tmp_group.clear();
        }
        tmp_group.push_back(tmp_op);
        base_groups.push_back(tmp_group);
        tmp_group.clear();
      }
    }
    group = tmp_group;
    is_binary = false;
  }
}

void GroupMethod::get_base_groups(
    std::vector<std::vector<Operation *>> &base_groups,
    const SetVector<Operation *> &subnet_ops) {
  std::vector<Operation *> group;
  bool is_binary = false;
  for (auto op : subnet_ops) {
    if (isLgSupport(op)) {
      if (!is_binary)
        is_binary = is_binary_shape_value(op);
      group.push_back(op);
      tmp_group_into_base(base_groups, group, op, is_binary);
    } else {
      if (!group.empty()) {
        base_groups.push_back(group);
        group.clear();
      }
      group.push_back(op);
      base_groups.push_back(group);
      group.clear();
      is_binary = false;
    }
  }

  if (!group.empty()) {
    base_groups.push_back(group);
  }
}

void GroupMethod::get_base_branch_groups(
    std::vector<std::vector<Operation *>> &base_groups,
    const SetVector<Operation *> &subnet_ops,
    const std::vector<Value> &subnet_return_opds) {
  std::vector<std::vector<Operation *>> tmp_base_groups;
  for (auto v : subnet_return_opds) { // ���������value��ʼ����
    auto tmp_op = v.getDefiningOp();
    std::vector<Operation *> tmp;
    tmp.push_back(tmp_op);
    tmp_base_groups.push_back(tmp);
  }

  llvm::errs() << "get_base_branch_groups start, group num:"
               << tmp_base_groups.size() << "\n";
  while (true) {
    bool can_break = true;
    for (auto group : tmp_base_groups) { // �ж��Ƿ����к�ѡ���Ѵ������
      if (group.back() != nullptr) {
        can_break = false;
        break;
      }
    }
    if (can_break) {
      break; // �Ѵ�����ϣ��˳�ѭ��
    }

    for (auto &group : tmp_base_groups) {
      auto tmp_op = group.back();
      if (tmp_op == nullptr) {
        continue;
      }
      int count = 0, imm_tensor_idx = 0, idx = 0;
      for (auto v : tmp_op->getOperands()) {
        auto pre_op = v.getDefiningOp();
        if (pre_op == nullptr ||
            isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
          continue;
        }
        if (isPreOpHaveAComputeOp(pre_op)) {
          count++;
          imm_tensor_idx = idx;
        }
        idx++;
      }
      llvm::errs() << "op:" << module::getName(tmp_op).str() << " have "
                   << count << " input tensor is not weight\n";
      if (count == 1) {
        auto tmp_op2 = tmp_op->getOperand(imm_tensor_idx).getDefiningOp();
        int user_count = 0;
        for (auto itr : tmp_op2->getResult(0).getUsers()) {
          if (!isa<ReturnOp>(itr)) {
            user_count++;
          }
        }
        llvm::errs() << "have " << user_count << " next node\n";
        if (user_count > 1) { // �����ֲ��
          group.push_back(nullptr);
          bool grp_exist = false;
          for (auto tmp_group : tmp_base_groups) {
            if (tmp_op2 == tmp_group[0]) {
              grp_exist = true;
              break;
            }
          }
          if (!grp_exist) {
            llvm::errs() << "meet divide node, add new group, start op name:"
                         << module::getName(tmp_op2).str() << "\n";
            std::vector<Operation *> tmp;
            tmp.push_back(tmp_op2);
            tmp_base_groups.push_back(tmp);
          }
        } else {
          group.push_back(tmp_op2);
        }
      } else if (count > 1) {     // ������ϵ�
        group.push_back(nullptr); // ������ǰ��ѡ��
        for (auto v : tmp_op->getOperands()) { // ����ϵ�����2���·�֧��Ϊ�µĺ�ѡ��
          auto pre_op = v.getDefiningOp();
          if (pre_op != nullptr &&
              isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
            continue;
          }
          llvm::errs() << "meet merge node, add new group, start op name:"
                       << module::getName(pre_op).str() << "\n";
          std::vector<Operation *> tmp;
          tmp.push_back(pre_op);
          tmp_base_groups.push_back(tmp);
        }
        break; // �������base_groups����Ԫ�أ���������ѭ��������Ӱ��ѭ������
      } else {
        group.push_back(nullptr);
        break; // �����������
      }
    }
  }

  for (auto &group : tmp_base_groups) {
    group.pop_back();
    reverse(group.begin(), group.end());
  }

  int i = 0;
  for (auto group : tmp_base_groups) {
    llvm::errs() << ">>>tmp_base_groups grp:" << i++ << "\n";
    int j = 0;
    for (auto op : group) {
      llvm::errs() << "  op:" << j++ << " name: " << module::getName(op).str()
                   << "\n";
    }
  }

  base_groups.clear();
  for (auto group : tmp_base_groups) {
    std::vector<Operation *> tmp;
    for (auto op : group) {
      if (isLgSupport(op)) {
        tmp.push_back(op);
      } else {
        llvm::errs() << "global layer name: " << module::getName(op).str()
                     << "\n";
        if (tmp.size() > 1) {
          base_groups.push_back(tmp);
        }
        tmp.clear();
      }
    }
    if (tmp.size() > 1) {
      base_groups.push_back(tmp);
    }
  }

  i = 0;
  for (auto group : base_groups) {
    llvm::errs() << ">>>base_groups grp:" << i++ << "\n";
    int j = 0;
    for (auto op : group) {
      llvm::errs() << "  op:" << j++ << " name: " << module::getName(op).str()
                   << "\n";
    }
  }
}

static void topo_order_dfs(Operation *cur_op, std::vector<Operation *> ops,
                           std::map<Operation *, int> &indeg,
                           std::vector<Operation *> &topo_ops) {
  topo_ops.push_back(cur_op);
  for (auto user : cur_op->getUsers()) {
    if (std::find(ops.begin(), ops.end(), user) != ops.end()) {
      indeg[user] = indeg[user] - 1;
      if (indeg[user] == 0) {
        if (std::find(topo_ops.begin(), topo_ops.end(), user) ==
            topo_ops.end()) {
          topo_order_dfs(user, ops, indeg, topo_ops);
        }
      }
    }
  }
}

static void find_op_tree_by_root(Operation *op,
                                 std::vector<Operation *> &op_tree,
                                 std::vector<Operation *> ops) {
  op_tree.push_back(op);
  for (auto user : op->getUsers()) {
    if (std::find(ops.begin(), ops.end(), user) != ops.end()) {
      find_op_tree_by_root(user, op_tree, ops);
    }
  }
}

void GroupMethod::get_base_dfs_topo_groups(
    std::vector<std::vector<Operation *>> &base_groups,
    const SetVector<Operation *> &subnet_ops,
    const std::vector<std::vector<Operation *>> &tmp_base_groups) {
  int i = 0;
  for (auto group : tmp_base_groups) {
    llvm::errs() << ">>>tmp_base_groups grp:" << i++ << "\n";
    int j = 0;
    for (auto op : group) {
      llvm::errs() << "  op:" << j++ << " name: " << module::getName(op).str()
                   << "\n";
    }
  }

  std::vector<LgInfo> LgInfos;
  for (auto group : tmp_base_groups) {
    LgInfo lg_info;
    lg_info.group_ops.assign(group.begin(), group.end());
    // lg_info.update_group_io(true);
    lg_info.update_group_io(LgPass::OPTIONS.opt);
    LgInfos.push_back(lg_info);
  }

  int idx = 0;
  for (auto LgInfo : LgInfos) {
    idx++;
    if (LgInfo.group_ops.size() == 1) {
      continue;
    }
    llvm::errs() << "start refine order, grp:" << --idx << "\n";
    std::vector<Operation *> topo_ops;
    std::map<Operation *, int> indeg;
    auto ops = LgInfo.group_ops;
    for (auto op : ops) {
      indeg[op] = 0;
      for (auto v : op->getOperands()) {
        if (std::find(ops.begin(), ops.end(), v.getDefiningOp()) != ops.end()) {
          if (indeg.find(op) != indeg.end()) {
            indeg[op] = indeg[op] + 1;
          }
        }
      }
    }
    std::vector<std::vector<Operation *>> serprated_groups;
    for (auto it : indeg) {
      llvm::errs() << "  indeg, op name:" << module::getName(it.first).str()
                   << ", count:" << it.second << "\n";
      if (it.second == 0) {
        if (std::find(topo_ops.begin(), topo_ops.end(), it.first) ==
            topo_ops.end()) {
          topo_order_dfs(it.first, ops, indeg, topo_ops);
        }

        std::vector<Operation *> op_tree;
        find_op_tree_by_root(it.first, op_tree,
                             ops); // ����ͼ���룬���ﶨ���ҵ�����ͼ��op_tree
        serprated_groups.push_back(op_tree);
      }
    }

    int i = 0;
    llvm::errs() << "full_topo_ops:\n";
    for (auto op : topo_ops) {
      llvm::errs() << "  op:" << i++ << " name:" << module::getName(op).str()
                   << "\n";
    }
    std::vector<std::vector<Operation *>> serprated_groups_checked;
    for (auto group : serprated_groups) { // ȷ����ͼ֮���Ƿ����
      bool separate = true;
      for (auto op : group) {
        for (auto group2 : serprated_groups) {
          if (group != group2 &&
              std::find(group2.begin(), group2.end(), op) != group2.end()) {
            separate = false;
            break;
          }
        }
        if (!separate) {
          break;
        }
      }
      if (separate) {
        serprated_groups_checked.push_back(group);
      }
    }

    for (auto group : serprated_groups_checked) { // ������topoͼ�л�ȡ��ֲ�topoͼ
      std::vector<Operation *> topo_group;
      for (auto op : topo_ops) {
        if (std::find(group.begin(), group.end(), op) != group.end()) {
          topo_group.push_back(op);
        }
      }
      base_groups.push_back(topo_group);
      llvm::errs() << "add new serprated_groups:\n";
      for (auto op : topo_group) {
        llvm::errs() << "  name:" << module::getName(op).str() << "\n";
      }
    }

    // ������topoͼ��ɾ������Ľڵ㣬ʣ�µ�ͼ�����������
    for (auto group : serprated_groups_checked) {
      for (auto op : group) {
        topo_ops.erase(std::remove(topo_ops.begin(), topo_ops.end(), op),
                       topo_ops.end());
      }
    }
    base_groups.push_back(topo_ops);
  }
}

static bool group_type_check(const LgInfo &lg_info) {
  auto group_type = lg_info.type;
  for (auto op : lg_info.group_ops) {
    if (isa<MatMulOp>(op)) {
      auto ins = op->getOperands();
      auto Lshape = module::getShape(ins[0]);
      int left_num_dims = Lshape.size();
      int right_num_dims = module::getShape(ins[1]).size();
      if (((left_num_dims == 4 && Lshape[1] < Lshape[2]) ||
           (left_num_dims == 5 && Lshape[1] < Lshape[3])) &&
          right_num_dims == 2) {
        if (group_type != GROUP_SMALL_C) {
          return false;
        }
      }
    }
  }
  return true;
}

static bool group_cslice_check(const LgInfo &lg_info) {
  if (module::isBM1684Family()) {
    for (auto op : lg_info.group_ops) {
      if (isa<ActiveOp>(op)) {
        auto shape = module::getShape(op->getOperand(0));
        if (shape.size() > 2 && shape[1] > 4096) {
          return false;
        }
      }
    }
  }
  return true;
}

bool GroupMethod::dynamic_group_valid_check(const LgInfo &lg_info) {
  auto res = true;
  if (runmode_ == RunMode::TPU_DYNAMIC && lg_info.group_ops.size() > 1) {
    // Condition 1
    // Dynamic Backend will choose the first op's batch as the whole group's
    // batch Need make sure dynamic group's ops have the same batch
    int64_t group_n =
        module::getShape(get_output_values(lg_info.group_ops[0])[0])[0];
    for (auto op : lg_info.group_ops) {
      if (!res)
        break;
      if (isa<tpu::ReshapeOp>(op)) {
        auto reshape_op = dyn_cast<tpu::ReshapeOp>(op);
        auto shape = module::getI64Array(reshape_op.getShape());
        for (auto s : *shape) {
          if (s < 0) {
            res = false;
            break;
          }
        }
      }
      auto outs = get_output_values(op);
      for (auto out : outs) {
        if (group_n != module::getShape(out)[0]) {
          res = false;
          break;
        }
      }
    }
    // Condition 2
    // Inputs and outputs number of a group cannot be large,
    // because it will cause much time to get info of inputs and outputs
    // when dynamic runtime. Also the MCU memory will not be enough
    // to store in/out node.
    const uint32_t max_io_num = 96;
    if (lg_info.group_ins.size() > max_io_num ||
        lg_info.group_outs.size() > max_io_num) {
      res = false;
    }
  }
  return res;
}

bool GroupMethod::group_valid_pre_check(const LgInfo &lg_info) {
  if (!group_type_check(lg_info)) {
    return false;
  }
  if (!group_cslice_check(lg_info)) {
    return false;
  }
  return true;
}

bool GroupMethod::is_layer_group_valid(LgInfo &lg_info, bool calc_cost,
                                       int64_t *group_cost) {
  bool status;
  status = group_one_layer_proc(lg_info, calc_cost, group_cost);
  if (status && LgPass::OPTIONS.group_by_cores == false) {
    return true;
  }

  if (!group_valid_pre_check(lg_info)) {
    return false;
  }

  shape_secs_t shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;
  if (!init_group_data_secs(lg_info, shape_secs, value_size)) {
    return false;
  }

  if (!dynamic_group_valid_check(lg_info)) {
    return false;
  }

  auto time_step = std::make_shared<BasicTimeStep>();
  status = time_step->assignTimeStep(lg_info, shape_secs, true);
  if (status == false) {
    return false;
  }

  auto lmem_allocator = std::make_shared<LmemAllocator>();
  status =
      lmem_allocator->assignLmemAddrWithSecs(lg_info, time_step, shape_secs);
  if (status == false) {
    return false;
  }

  if (calc_cost) {
// remove it after pid_node is extractedb
#pragma omp critical(get_cycle)
    *group_cost =
        cycle_calculator_->getGroupCycle(time_step, shape_secs, lg_info.type);
  }
  // llvm::errs() << "nsecs = " << shape_secs.nsecs
  //              << ", hsecs = " << shape_secs.hsecs << "\n";
  return status;
}

void GroupMethod::get_layer_cut_result(
    std::vector<int64_t> &cut_result,
    const std::vector<std::pair<int64_t, int64_t>> &clusters,
    const std::vector<std::vector<int64_t>> &cut_points, int64_t start,
    int64_t end) {
  int64_t opt_cut = cut_points[start][end];
  if (opt_cut != end) {
    get_layer_cut_result(cut_result, clusters, cut_points, start, opt_cut);
    get_layer_cut_result(cut_result, clusters, cut_points, opt_cut + 1, end);
  } else {
    cut_result.push_back(clusters[end].first + clusters[end].second - 1);
  }
}

void GroupMethod::get_group_clusters(
    std::vector<std::pair<int64_t, int64_t>> &clusters,
    const std::vector<Operation *> &base_group) {
  LgInfo sub_group;
  size_t group_layer_num = base_group.size();
  const int64_t max_cluster_size = get_max_cluster_size(group_layer_num);
  int64_t start_idx = 0, end_idx = 1, cluster_size = 1;
  if (max_cluster_size > 1) {
    int64_t pre_cost = 0;
    for (size_t idx = 1; idx < group_layer_num; ++idx) {
      if (start_idx == end_idx - 1) {
        pre_cost =
            cycle_calculator_->getGlobalLayerCycle(base_group[start_idx]);
      }
      pre_cost += cycle_calculator_->getGlobalLayerCycle(base_group[end_idx]);

      int64_t temp_cost = 0;
      get_layer_group(sub_group, base_group, start_idx, end_idx);
      bool is_valid = is_layer_group_valid(sub_group, true, &temp_cost);
      if (is_valid) {
        if (pre_cost <= temp_cost) {
          is_valid = false;
        } else {
          pre_cost = temp_cost;
        }
      }

      if (!is_valid || (is_valid && cluster_size >= max_cluster_size - 1) ||
          idx == group_layer_num - 1) {
        if (is_valid) {
          ++cluster_size;
        }
        clusters.push_back(std::make_pair(start_idx, cluster_size));
        start_idx = is_valid ? end_idx + 1 : end_idx;
        idx = is_valid ? idx + 1 : idx;
        end_idx = start_idx + 1;
        cluster_size = 1;
        pre_cost = 0;
        if ((!is_valid && idx == group_layer_num - 1) ||
            start_idx == group_layer_num - 1) {
          clusters.push_back(std::make_pair(start_idx, cluster_size));
          if (start_idx == group_layer_num - 1) {
            break;
          }
        }
      } else {
        ++cluster_size;
        ++end_idx;
      }
    }
  } else {
    for (size_t layer_idx = 0; layer_idx < group_layer_num; ++layer_idx) {
      clusters.push_back(std::make_pair<int64_t, int64_t>(layer_idx, 1));
    }
  }

  llvm::errs() << "clusters idx(size): ";
  for (size_t i = 0; i < clusters.size(); ++i) {
    llvm::errs() << llvm::format("%d(%d), ", clusters[i].first,
                                 clusters[i].second);
  }
  llvm::errs() << "\n";
}

void GroupMethod::sweep_for_min_cost(
    int64_t *group_cost, int64_t *optimal_point, int64_t start, int64_t end,
    const std::vector<std::vector<int64_t>> &cost_table) {
  for (int64_t sweep = start; sweep < end; ++sweep) {
    int64_t temp_cost =
        cost_add(cost_table[start][sweep], cost_table[sweep + 1][end]);
    if (temp_cost < *group_cost) {
      *group_cost = temp_cost;
      *optimal_point = sweep;
    }
  }
}

void createSubnetGraph(std::vector<Operation *> subnet_ops,
                       std::shared_ptr<dot_graph> dot_graph_log) {
  for (auto op : subnet_ops) {
    if (!isa<ReturnOp>(op)) {
      auto op_name = module::getName(op).str() + "_ori";
      dot_graph_log->add_node_into_graph(op_name);
      dot_graph_log->add_node_label(op_name,
                                    op->getName().getStringRef().str());

      bool next_layer_has_return = false;
      for (auto itr = op->user_begin(); itr != op->user_end(); itr++) {
        if (!isa<ReturnOp>(*itr)) {
          auto to = module::getName(*itr).str() + "_ori";
          dot_graph_log->add_node_into_graph(to);
          dot_graph_log->add_node_label(to,
                                        (*itr)->getName().getStringRef().str());
          dot_graph_log->add_edge_into_graph(op_name, to);
        } else {
          next_layer_has_return = true;
        }
      }
      if (next_layer_has_return) {
        dot_graph_log->add_node_label(op_name, "to_returnOp");
      }
    }
  }
}

void GroupMethod::dynamic_programming_layer_group_with_cluster(
    std::vector<LgInfo> &lg_infos, const SetVector<Operation *> &subnet_ops) {
  llvm::errs() << "\n"
               << "=======================================================\n"
               << "***** Dynamic Programming layer group with cluster ****\n"
               << "=======================================================\n";
  // for debug
  // std::vector<Operation *> ops_vector;
  // for (Operation *op : subnet_ops) {
  //       ops_vector.push_back(op);
  // }
  // std::shared_ptr<dot_graph> opt2_dot_graph = std::make_shared<dot_graph>();
  // createSubnetGraph(ops_vector, opt2_dot_graph);
  // for debug
  LgInfo sub_group;
  std::vector<std::vector<Operation *>> base_groups;
  get_base_groups(base_groups, subnet_ops);
  llvm::errs() << llvm::format("total num of base_group is %d\n",
                               base_groups.size());
  for (size_t i = 0; i < base_groups.size(); ++i) {
    std::vector<std::pair<int64_t, int64_t>> clusters;
    get_group_clusters(clusters, base_groups[i]);
    size_t cluster_num = clusters.size();
    llvm::errs() << llvm::format(
        "process base group %d, layer_num=%d, cluster_num=%d\n", i,
        base_groups[i].size(), cluster_num);
    if (cluster_num > 1) {
      auto cost_table = std::vector<std::vector<int64_t>>(
          cluster_num, std::vector<int64_t>(cluster_num, 0));
      auto cut_points = std::vector<std::vector<int64_t>>(
          cluster_num, std::vector<int64_t>(cluster_num, 0));
      for (size_t j = 0; j < cluster_num; ++j) {
        int64_t start_idx = clusters[j].first;
        int64_t end_idx = start_idx + clusters[j].second - 1;
        get_layer_group(sub_group, base_groups[i], start_idx, end_idx);

        assert(is_layer_group_valid(sub_group, true, &cost_table[j][j]));

        LLVM_DEBUG({
          llvm::errs() << "cluster[" << j << "] = " << start_idx << ", "
                       << end_idx << ";" << "cost = " << cost_table[j][j]
                       << "\n";
          sub_group.dump_lginfo();
        });

        cut_points[j][j] = j;
      }
      llvm::errs() << "Searching best group slices...\n";
      progressbar bar(cluster_num - 1);
      for (size_t len = 2; len <= cluster_num; ++len) {
        bar.update();
        // llvm::errs() << llvm::format("process cluster len = %d\n", len);
        // #pragma omp parallel for private(sub_group)
        for (int64_t start = 0; start <= cluster_num - len; ++start) {
          int64_t end = start + len - 1;
          // llvm::errs() << "start = " << start << ", end = " << end << "\n";
          int64_t start_idx = clusters[start].first;
          int64_t end_idx = clusters[end].first + clusters[end].second - 1;
          get_layer_group(sub_group, base_groups[i], start_idx, end_idx);

          int64_t group_cost = MAX_COST;
          is_layer_group_valid(sub_group, true, &group_cost);

          int64_t optimal_point = end;
          // sweep_for_min_cost(&group_cost, &optimal_point, start, end,
          //                    cost_table);
          LLVM_DEBUG({
            llvm::errs() << "; start_idx = " << start_idx
                         << "; end_idx = " << end_idx
                         << "; group_cost = " << group_cost << "\n";
          });

          for (int64_t sweep = start; sweep < end; ++sweep) {
            int64_t temp_cost =
                cost_add(cost_table[start][sweep], cost_table[sweep + 1][end]);
            if (temp_cost < group_cost) {
              group_cost = temp_cost;
              optimal_point = sweep;
              LLVM_DEBUG({
                llvm::errs() << "; update better" << "; start = " << start
                             << "; sweep = " << sweep << "; end = " << end
                             << "; temp_cost = " << temp_cost << "\n";
              });
            }
          }
          LLVM_DEBUG({
            llvm::errs() << "; start_idx = " << start_idx
                         << "; end_idx = " << end_idx
                         << "; group_cost = " << group_cost << "\n";
          });

          cost_table[start][end] = group_cost;
          cut_points[start][end] = optimal_point;
        }
      }
      llvm::errs() << "\n";
      std::vector<int64_t> cut_result;
      get_layer_cut_result(cut_result, clusters, cut_points, 0,
                           cluster_num - 1);
      cut_results_.push_back(std::move(cut_result));
      LLVM_DEBUG({
        LgInfo sub_group;
        int start = 0;
        for (auto end : cut_result) {
          get_layer_group(sub_group, base_groups[i], start, end);
          int64_t group_cost = MAX_COST;
          auto temp_status = is_layer_group_valid(sub_group, true, &group_cost);
          llvm::errs() << temp_status << " ;start" << start << " - " << " end "
                       << end << " = " << group_cost << "\n";
          start = end + 1;
        }

        llvm::errs() << "\n";
        llvm::errs() << "================FINAL GROUP================\n";
        for (size_t cost_i = 0; cost_i < cluster_num; ++cost_i) {
          for (int64_t cost_j = 0; cost_j < cluster_num; ++cost_j) {
            llvm::errs() << cut_points[cost_i][cost_j] << ", " << "";
          }
          llvm::errs() << "\n";
        }
        llvm::errs() << "================COST TABLE================\n";
        for (size_t cost_i = 0; cost_i < cluster_num; ++cost_i) {
          for (int64_t cost_j = 0; cost_j < cluster_num; ++cost_j) {
            llvm::errs() << cost_table[cost_i][cost_j] << ", " << "";
          }
          llvm::errs() << "\n";
        }
        llvm::errs() << "=============================================\n";
        llvm::errs() << "\n";
      });
    } else {
      cut_results_.push_back(std::vector<int64_t>(1, 0));
    }
  }

  show_cut_results();
  // some post process for cluster
  llvm::errs() << "-------------------------------------------------------\n";
  llvm::errs() << "Consider redundant computation and gdma cost\n";
  llvm::errs() << "-------------------------------------------------------\n";
  consider_redundant_computation_and_gdma_cost(base_groups);
  show_cut_results();

  llvm::errs() << "-------------------------------------------------------\n";
  llvm::errs() << "Merge cut idx to reduce gdma cost\n";
  llvm::errs() << "-------------------------------------------------------\n";
  bool take_effective = merge_cut_idx_to_reduce_gdma_cost(base_groups);
  show_cut_results();

  if (take_effective) {
    llvm::errs() << "-------------------------------------------------------\n";
    llvm::errs() << "Consider redundant computation and gdma cost again\n"
                 << "due to cut idx merged in the previous step\n";
    llvm::errs() << "-------------------------------------------------------\n";
    consider_redundant_computation_and_gdma_cost(base_groups);
    show_cut_results();
  }

  // update lg_infos
  get_final_groups(lg_infos, base_groups);
  // for debug
  // int grp_idx = 0;
  // for (auto lg_info : lg_infos) {
  //   if(lg_info.group_ops.size()>1){
  //     for (auto op : lg_info.group_ops) {
  //       if(!isa<ReturnOp>(op)){
  //         auto name = module::getName(op).str();
  //         opt2_dot_graph->add_node_label(name + "_ori",
  //                                       "grp_" + std::to_string(grp_idx));
  //       }
  //     }
  //     grp_idx++;
  //   }
  // }
  // std::cout<<"attention !!! opt2 grp"<<grp_idx<<std::endl;
  // opt2_dot_graph->export_dot("opt2_ok");
  // for debug
}

bool GroupMethod::update_sequence_group_cost(LgInfo *left_layer_group,
                                             LgInfo *right_layer_group,
                                             bool *left_first,
                                             SequenceGroupsInfo &opt_seq_info) {
  assert(left_layer_group->group_ops.size() > 0);
  assert(right_layer_group->group_ops.size() > 0);
  LgInfo *groups[2];
  shape_secs_t *p_shape_secs[2];
  if (*left_first) {
    groups[0] = left_layer_group;
    groups[1] = right_layer_group;
    p_shape_secs[0] = &(opt_seq_info.left_shape_secs);
    p_shape_secs[1] = &(opt_seq_info.right_shape_secs);
  } else {
    groups[0] = right_layer_group;
    groups[1] = left_layer_group;
    p_shape_secs[0] = &(opt_seq_info.right_shape_secs);
    p_shape_secs[1] = &(opt_seq_info.left_shape_secs);
  }
  bool valid = true;
  shape_secs_t shape_secs[2];
  BasicTimeStepPtr time_steps[2] = {std::make_shared<BasicTimeStep>(),
                                    std::make_shared<BasicTimeStep>()};
  auto lmem_allocator = std::make_shared<LmemAllocator>();
  int64_t group_costs[2] = {0, 0};
  bool pre_cost_judge = true;
  for (size_t i = 0; i < 2; ++i) {
    if (group_one_layer_proc(*groups[i], true, &group_costs[i])) {
      shape_secs[i].nsecs = 1;
      shape_secs[i].csecs = 1;
      shape_secs[i].hsecs = 1;
      shape_secs[i].dsecs = 1;
      shape_secs[i].wsecs = 1;
      continue;
    }

    std::vector<std::pair<Value, int64_t>> value_size;
    if (!init_group_data_secs(*groups[i], shape_secs[i], value_size)) {
      valid = false;
      break;
    }
    if (!time_steps[i]->assignTimeStep(*groups[i], shape_secs[i], true)) {
      valid = false;
      break;
    }
    if (!update_data_split(time_steps[i], *groups[i], shape_secs[i])) {
      valid = false;
      break;
    }

    *left_first = !(*left_first);
    if (pre_cost_judge) {
      if (memcmp(&shape_secs[i], p_shape_secs[i], sizeof(shape_secs_t)) != 0) {
        pre_cost_judge = false;
        continue;
      }
      if (!stripe_mine_max_slice(*groups[i], shape_secs[i],
                                 time_steps[i]->get_tensor_infos())) {
        valid = false;
        break;
      }
      group_costs[i] = cycle_calculator_->getGroupCycle(
          time_steps[i], shape_secs[i], groups[i]->type);
    }
  }
  if (!valid) {
    return false;
  }
  int64_t total_cost = group_costs[0] + group_costs[1];
  if (pre_cost_judge) {
    LLVM_DEBUG(llvm::errs() << "The pre cost of the two group is " << total_cost
                            << "\n";);
    if (opt_seq_info.min_cost >= 0 && opt_seq_info.min_cost < total_cost) {
      return false;
    }
  }

  for (size_t i = 0; i < 2; ++i) {
    if (groups[i]->group_ops.size() == 1) {
      continue;
    }
    if (!lmem_allocator->assignLmemAddrWithSecs(*groups[i], time_steps[i],
                                                shape_secs[i])) {
      valid = false;
      break;
    }
    *left_first = !(*left_first);
    group_costs[i] = cycle_calculator_->getGroupCycle(
        time_steps[i], shape_secs[i], groups[i]->type);
  }
  if (!valid) {
    return false;
  }
  total_cost = group_costs[0] + group_costs[1];
  llvm::errs() << "The final cost of the two group is " << total_cost << "\n";
  if (opt_seq_info.min_cost >= 0 && opt_seq_info.min_cost <= total_cost) {
    return false;
  }
  opt_seq_info.min_cost = total_cost;
  memcpy(p_shape_secs[0], &shape_secs[0], sizeof(shape_secs_t));
  memcpy(p_shape_secs[1], &shape_secs[1], sizeof(shape_secs_t));

  return true;
}

bool GroupMethod::consider_redundant_computation_and_gdma_cost(
    const std::vector<std::vector<Operation *>> &base_groups) {

  int64_t left_cut_idx;
  int64_t optimal_cut_idx;
  SequenceGroupsInfo seq_info;
  LgInfo left_sub_group, right_sub_group;

  for (size_t i = 0; i < base_groups.size(); ++i) {
    auto &base_group = base_groups[i];
    auto &cut_result = cut_results_[i];
    size_t cut_num = cut_result.size();
    if (cut_num > 1 && get_max_cluster_size(base_group.size()) > 1) {
      for (int32_t j = cut_num - 2; j >= 0; --j) {
        left_cut_idx = j > 0 ? (cut_result[j - 1] + 1) : (int64_t)0;

        memset(&seq_info, 0, sizeof(SequenceGroupsInfo));
        seq_info.min_cost = -1;
        optimal_cut_idx = cut_result[j];
        cut_result[j] = cut_result[j + 1] - 1;
        bool left_first = true;
        for (; cut_result[j] >= left_cut_idx; cut_result[j]--) {
          get_layer_group(left_sub_group, base_group, left_cut_idx,
                          cut_result[j]);
          get_layer_group(right_sub_group, base_group, cut_result[j] + 1,
                          cut_result[j + 1]);
          bool is_better = update_sequence_group_cost(
              &left_sub_group, &right_sub_group, &left_first, seq_info);
          if (is_better) {
            optimal_cut_idx = cut_result[j];
            llvm::errs() << "//// Group cost " << seq_info.min_cost
                         << ", optimal cut idx " << optimal_cut_idx << "\n";
          }
        }
        cut_result[j] = optimal_cut_idx;
      }
    }
  }
  return true;
}

bool GroupMethod::merge_cut_idx_to_reduce_gdma_cost(
    const std::vector<std::vector<Operation *>> &base_groups) {
  LgInfo sub_group;
  bool lg_valid;
  bool take_effective = false;
  for (size_t i = 0; i < base_groups.size(); ++i) {
    auto &base_group = base_groups[i];
    auto &cut_result = cut_results_[i];
    if (get_max_cluster_size(base_group.size()) > 1) {
      int64_t left_group_cost = 0, right_group_cost = 0;
      int64_t combine_group_cost = 0;
      size_t size_ = cut_result.size();
      for (size_t j = 0; j < size_ - 1;) {
        size_t cut_idx = cut_result[j];
        size_t start_cut_idx = j > 0 ? (cut_result[j - 1] + 1) : 0;
        size_t end_cut_idx = cut_result[j + 1];
        // get left sub_group
        if (left_group_cost == 0) {
          get_layer_group(sub_group, base_group, start_cut_idx, cut_idx);
          lg_valid = is_layer_group_valid(sub_group, true, &left_group_cost);
          assert(lg_valid);
        }
        // get right sub_group
        get_layer_group(sub_group, base_group, cut_idx + 1, end_cut_idx);
        lg_valid = is_layer_group_valid(sub_group, true, &right_group_cost);
        assert(lg_valid);

        // get combine group
        get_layer_group(sub_group, base_group, start_cut_idx, end_cut_idx);
        lg_valid = is_layer_group_valid(sub_group, true, &combine_group_cost);
        if (lg_valid) {
          if (combine_group_cost < left_group_cost + right_group_cost) {
            LLVM_DEBUG({
              llvm::errs() << "; start_idx = " << start_cut_idx
                           << "; end_idx = " << end_cut_idx
                           << "; group_cost = " << combine_group_cost
                           << "; base_group = " << i << "; action = "
                           << "merge_cut_idx_to_reduce_gdma_cost" << "\n";
            });
            cut_result.erase(cut_result.begin() + j);
            size_ = cut_result.size();
            take_effective = true;
            left_group_cost = combine_group_cost;
          } else {
            j++;
            left_group_cost = right_group_cost;
          }
        } else {
          j++;
          left_group_cost = right_group_cost;
        }
      }
    }
  }
  return take_effective;
}

void GroupMethod::simple_layer_group(std::vector<LgInfo> &lg_infos,
                                     const SetVector<Operation *> &subnet_ops) {
  llvm::errs() << "\n"
               << "=======================================================\n"
               << "*********** Group layers as many as possible **********\n"
               << "=======================================================\n";

  cut_results_.clear();
  LgInfo sub_group;
  std::vector<std::vector<Operation *>> base_groups;
  get_base_groups(base_groups, subnet_ops);
  for (int64_t i = base_groups.size() - 1; i >= 0; --i) {
    std::vector<int64_t> cut_result;
    if (base_groups[i].size() == 1) {
      cut_result.push_back(0);
      cut_results_.insert(cut_results_.begin(), std::move(cut_result));
      continue;
    }
    int64_t start_idx = 0, end_idx = base_groups[i].size() - 1;
    cut_result.insert(cut_result.begin(), end_idx);
    while (end_idx > start_idx) {
      get_layer_group(sub_group, base_groups[i], start_idx, end_idx);
      bool valid = is_layer_group_valid(sub_group, false, nullptr);
      if (valid) {
        if (start_idx > 0) {
          cut_result.insert(cut_result.begin(), start_idx - 1);
          end_idx = start_idx - 1;
          start_idx = 0;
        } else {
          break;
        }
      } else {
        start_idx++;
        if (start_idx == end_idx) {
          cut_result.insert(cut_result.begin(), start_idx - 1);
          end_idx = start_idx - 1;
          start_idx = 0;
        }
      }
    }
    cut_results_.insert(cut_results_.begin(), std::move(cut_result));
  }
  show_cut_results();
  get_final_groups(lg_infos, base_groups);
}

static inline int64_t increase_nsecs2(int64_t nsecs, int64_t batch_size) {
  if (nsecs == batch_size) {
    return -1;
  }
  int64_t nslice = batch_size / nsecs + (batch_size % nsecs > 0);
  int64_t new_nslice = nslice;
  int64_t next_nsecs = nsecs;
  do {
    next_nsecs++;
    new_nslice = batch_size / next_nsecs + (batch_size % next_nsecs > 0);
  } while (new_nslice >= nslice && next_nsecs < batch_size);

  return next_nsecs;
}

static inline int64_t increase_csecs2(int64_t csecs, int64_t max_csecs) {
  if (csecs == max_csecs) {
    return -1;
  }
  int64_t cslice = max_csecs / csecs + (max_csecs % csecs > 0);
  int64_t new_cslice = cslice;
  int64_t next_csecs = csecs;
  do {
    next_csecs++;
    new_cslice = max_csecs / next_csecs + (max_csecs % next_csecs > 0);
  } while (new_cslice >= cslice && next_csecs < max_csecs);

  return next_csecs;
}

static inline void update_shape_secs2(const LgInfo &lg_info,
                                      shape_secs_t &shape_secs,
                                      int64_t &dhw_secs,
                                      const shape_secs_t &max_shape_secs) {
  if (shape_secs.nsecs < max_shape_secs.nsecs) {
    shape_secs.nsecs = increase_nsecs2(shape_secs.nsecs, max_shape_secs.nsecs);
  } else if (shape_secs.csecs < max_shape_secs.csecs) {
    shape_secs.csecs = increase_csecs2(shape_secs.csecs, max_shape_secs.csecs);
  } else {
    assign_dhwsecs(lg_info, shape_secs, ++dhw_secs, max_shape_secs);
  }
}

static inline void update_shape_secs_for_ilp_group(shape_secs_t &shape_secs,const shape_secs_t &max_shape_secs) {

  //
  // if (shape_secs.nsecs < max_shape_secs.nsecs) {
  //   int64_t nsecs = shape_secs.nsecs;
  //   int64_t batch_size = max_shape_secs.nsecs;
  //   int64_t nslice = batch_size / nsecs + (batch_size % nsecs > 0);
  //   int64_t new_nslice = nslice;
  //   int64_t next_nsecs = nsecs;

  //   do {
  //     next_nsecs++;
  //     new_nslice = batch_size / next_nsecs + (batch_size % next_nsecs > 0);
  //   } while (new_nslice >= nslice && next_nsecs < batch_size);

  //   shape_secs.nsecs = next_nsecs;
  // } else if (shape_secs.hsecs < max_shape_secs.hsecs) {
  //   int64_t hsecs = shape_secs.hsecs;
  //   int64_t max_hsecs = max_shape_secs.hsecs;
  //   int64_t hslice = max_hsecs / hsecs + (max_hsecs % hsecs > 0);
  //   int64_t new_hslice = hslice;
  //   int64_t next_hsecs = hsecs;

  //   do {
  //     next_hsecs++;
  //     new_hslice = max_hsecs / next_hsecs + (max_hsecs % next_hsecs > 0);
  //   } while (new_hslice >= hslice && next_hsecs < max_hsecs);

  //     shape_secs.hsecs = next_hsecs;
  //   }

  //
  if (shape_secs.nsecs < max_shape_secs.nsecs) {
    shape_secs.nsecs++;
  } else if (shape_secs.hsecs < max_shape_secs.hsecs) {
    shape_secs.hsecs++;
  }

  return;
}

std::vector<int>
GroupMethod::get_sec_per_cores(const shape_secs_t &shape_secs,
                               std::vector<std::vector<int64_t>> &vec_ncdhw,
                               int core_num, TensorInfo &tensor_infos) {
  int secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs *
             shape_secs.hsecs * shape_secs.wsecs;
  int secs_per_core = secs / core_num;
  auto sec_per_cores = std::vector<int>();
  for (int i = 0; i < core_num; i++) {
    sec_per_cores.push_back(secs_per_core);
  }
  int rest = secs - core_num * secs_per_core;
  for (int i = 0; i < core_num; i++) {
    if (--rest < 0) {
      break;
    }
    sec_per_cores[i]++;
  }

  for (int i = 0; i < core_num; i++) {
    llvm::errs() << "sec_per_cores:" << sec_per_cores[i] << "\n";
  }

  for (int n = 0; n < shape_secs.nsecs; n++) { // todo Ѱ���ø���core����ˮһ�µ�˳��
    for (int c = 0; c < shape_secs.csecs; c++) {
      for (int d = 0; d < shape_secs.dsecs; d++) {
        for (int h = 0; h < shape_secs.hsecs; h++) {
          for (int w = 0; w < shape_secs.wsecs; w++) {
            std::vector<int64_t> tmp;
            tmp.push_back(n);
            tmp.push_back(c);
            tmp.push_back(d);
            tmp.push_back(h);
            tmp.push_back(w);
            vec_ncdhw.push_back(tmp);
          }
        }
      }
    }
  }

  // struct vec_ncdhw_compare {
  //   bool operator()(std::vector<int> v0, std::vector<int> v1) const {
  //     for (int i = 0; i < v0.size(); i++) {
  //       if (v0[i] < v1[i]) {
  //         return true;
  //       }
  //     }
  //     return false;
  //   }
  // };

  // std::set<std::vector<int>, vec_ncdhw_compare> unique_ncdhw_set;
  // for (auto itr = tensor_infos.begin(); itr != tensor_infos.end(); ++itr) {
  //   if (itr->second.slice_info.n[itr1[0]] !=
  //   itr->second.slice_info.n[itr2[0]] ||
  //       itr->second.slice_info.c[itr1[1]] !=
  //       itr->second.slice_info.c[itr2[1]] ||
  //       itr->second.slice_info.d[itr1[2]] !=
  //       itr->second.slice_info.d[itr2[2]] ||
  //       itr->second.slice_info.h[itr1[3]] !=
  //       itr->second.slice_info.h[itr2[3]] ||
  //       itr->second.slice_info.w[itr1[4]] !=
  //       itr->second.slice_info.w[itr2[4]]) {
  //         return false;
  //   }
  // }

  // for (auto itr1 = vec_ncdhw.begin(); itr1 != vec_ncdhw.end(); ++itr) {
  //   for (auto itr2 = vec_ncdhw.begin(); itr2 != vec_ncdhw.end(); ++itr2) {
  //     for (auto itr = tensor_infos.begin(); itr != tensor_infos.end(); ++itr)
  //     {
  //       if (itr->second.slice_info.n[itr1[0]] !=
  //       itr->second.slice_info.n[itr2[0]] ||
  //           itr->second.slice_info.c[itr1[1]] !=
  //           itr->second.slice_info.c[itr2[1]] ||
  //           itr->second.slice_info.d[itr1[2]] !=
  //           itr->second.slice_info.d[itr2[2]] ||
  //           itr->second.slice_info.h[itr1[3]] !=
  //           itr->second.slice_info.h[itr2[3]] ||
  //           itr->second.slice_info.w[itr1[4]] !=
  //           itr->second.slice_info.w[itr2[4]]) {

  //       }
  //     }
  //   }
  // }

  // reverse(vec_ncdhw.begin(),vec_ncdhw.end());
  return std::move(sec_per_cores);
}

// std::vector<std::pair<int,int>> get_var_low_high_bound(int slice_num, int
// group_size, int overlap_size = 1) {
//   std::vector<std::pair<int,int>> tmp;
//   int end = slice_num*group_size + 1;
//   if (slice_num == 1) {
//     tmp.push_back(std::make_pair(0, end));
//   } else if (slice_num == 2) {
//     tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     tmp.push_back(std::make_pair(group_size - overlap_size + 1, end));
//   } else {
//     tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     for (int i = 0; i < slice_num - 2) {
//       tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     }
//     tmp.push_back(std::make_pair(0, 2*group_size + 1));
//   }
//   return std::move(tmp);
// }

// std::vector<std::pair<int,int>> get_var_low_high_bound(int slice_num, int
// group_size, int overlap_size = 1) {
//   std::vector<std::pair<int,int>> tmp;
//   int end = slice_num*group_size + 1;
//   if (slice_num == 1) {
//     tmp.push_back(std::make_pair(0, end));
//   } else if (slice_num == 2) {
//     tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     tmp.push_back(std::make_pair(group_size - overlap_size + 1, end));
//   } else {
//     tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     for (int i = 0; i < slice_num - 2) {
//       tmp.push_back(std::make_pair(0, group_size + 1 + overlap_size));
//     }
//     tmp.push_back(std::make_pair(0, 2*group_size + 1));
//   }
//   return std::move(tmp);
// }

std::vector<std::vector<Operation *>>
ExcludeOpFromGroup(std::vector<Operation *> &group_ops, Operation *fail_op) {
  std::vector<Operation *> pre_ops, next_ops;
  find_all_pre_ops(fail_op, pre_ops);
  find_all_next_ops(fail_op, next_ops);
  pre_ops.erase(std::remove(pre_ops.begin(), pre_ops.end(), fail_op),
                pre_ops.end());
  next_ops.erase(std::remove(next_ops.begin(), next_ops.end(), fail_op),
                 next_ops.end());

  std::vector<std::vector<Operation *>> new_grp;
  std::vector<Operation *> del_ops;
  for (auto op : group_ops) {
    if (std::find(pre_ops.begin(), pre_ops.end(), op) != pre_ops.end()) {
      for (auto op2 : group_ops) {
        if (std::find(next_ops.begin(), next_ops.end(), op2) !=
                next_ops.end() &&
            std::find(del_ops.begin(), del_ops.end(), op2) == del_ops.end()) {
          del_ops.push_back(op2);
        }
      }
    }
  }

  if (del_ops.size() > 0) {
    llvm::errs() << "get a new grp:\n";
  }
  for (auto del_op : del_ops) {
    llvm::errs() << "  name:" << module::getName(del_op).str() << "\n";
    group_ops.erase(std::remove(group_ops.begin(), group_ops.end(), del_op),
                    group_ops.end());
  }
  group_ops.erase(std::remove(group_ops.begin(), group_ops.end(), fail_op),
                  group_ops.end());
  if (del_ops.size() > 0) {
    new_grp.push_back(del_ops); // һ��groupֻ��ֳ�2�������Ƕ��??
  }
  return new_grp;
}

void processWhenOpFail(LgPassIR *pass_ir, LgInfo &sub_group,
                       std::vector<std::vector<Operation *>> &base_groups,
                       int &grp_num, Operation *fail_op) {
  auto new_grps = ExcludeOpFromGroup(sub_group.group_ops, fail_op);
  for (auto new_grp : new_grps) {
    grp_num++;
    base_groups.push_back(new_grp);
    pass_ir->ILP_time_steps.push_back(std::vector<ILPTimeStepPtr>());
    pass_ir->map_l2m_load.push_back(
        std::map<int, std::vector<l2m_value_info>>());
  }
  sub_group.update_group_io(LgPass::OPTIONS.opt);
  set_group_type(sub_group);
}

void mergeEdgeOpToOtherGrp(int cur_grp_idx,
                           std::vector<std::vector<Operation *>> &base_groups,
                           Operation *fail_op) {
  std::vector<Operation *> vec_near_op;
  for (auto user : fail_op->getUsers()) {
    if (!isa<ReturnOp>(user)) {
      vec_near_op.push_back(user);
    }
  }
  for (auto v : fail_op->getOperands()) {
    auto pre_op = v.getDefiningOp();
    if (pre_op && !isa<top::NoneOp>(pre_op)) {
      vec_near_op.push_back(pre_op);
    }
  }
  int max_group_idx = -1;
  for (
      auto it :
      vec_near_op) { // 寻找在del_op的各个方向,相邻的，有最多op的group，然后把del_op融合进去
    for (int j = cur_grp_idx + 1; j < base_groups.size();
         j++) { // 只融合到后面还未规划的group
      if (std::find(base_groups[j].begin(), base_groups[j].end(), it) !=
          base_groups[j].end()) {
        max_group_idx = std::max(max_group_idx, j);
      }
    }
  }
  if (max_group_idx != -1) {
    base_groups[max_group_idx].push_back(fail_op);
    llvm::errs() << "add fail_op to group" << max_group_idx << "\n";
  } else {
    llvm::errs() << "make fail_op as global layer\n";
  }
}

std::vector<op_var_pos_info> createOverlapStrategy(const LgInfo &lg_info,
                                                   int slice_num, int type = 0,
                                                   int overlap = 2,
                                                   int fix_gap = 4) {
  std::vector<op_var_pos_info> op_var_bound;
  op_var_pos_info null_var_pos;
  null_var_pos.ts_id = 0;
  op_var_bound.push_back(null_var_pos);
  int k = 1;
  int op_num = lg_info.group_ops.size();
  llvm::errs() << "old overlap:" << overlap << "\n";
  if (op_num <= overlap) {
    overlap = 1;
  } else if (op_num * 0.2 > overlap) {
    overlap = op_num * 0.2;
  }
  llvm::errs() << "new overlap:" << overlap << "\n";
  for (int n = 0; n < slice_num; n++) {
    int group_offset = k;
    for (int m = 0; m < op_num; m++) {
      op_var_pos_info var_pos;
      var_pos.ts_id = k++;
      var_pos.key = std::make_pair(n, m);
      if (type == 0) {
        var_pos.start_ts = group_offset - overlap;
        var_pos.end_ts = group_offset + op_num + overlap - 1;
      } else if (type == 1) {
        var_pos.start_ts = var_pos.ts_id - fix_gap;
        var_pos.end_ts = var_pos.ts_id + fix_gap;
      }

      if (var_pos.start_ts < 0) {
        var_pos.start_ts = 0;
      }
      if (var_pos.end_ts >= slice_num * op_num + 2) {
        var_pos.end_ts = slice_num * op_num + 1;
      }
      op_var_bound.push_back(var_pos);
    }
  }
  null_var_pos.ts_id = k;
  op_var_bound.push_back(null_var_pos);
  return std::move(op_var_bound);
}

void showTensorInfo(TensorInfo tensor_infos) {
  LOG(INFO) << "showTensorInfo:";
  for (auto itr = tensor_infos.begin(); itr != tensor_infos.end(); ++itr) {
    LOG(INFO) << " tensor name: " << module::getName(itr->first).str();
    int i = 0;
    for (auto itr2 : itr->second.slice_info.n) {
      LOG(INFO) << "  n: " << i << " (" << itr2.first << "," << itr2.second
                << ")";
      i++;
    }
    i = 0;
    for (auto itr2 : itr->second.slice_info.h) {
      LOG(INFO) << "  h: " << i << " (" << itr2.first << "," << itr2.second
                << ")";
      i++;
    }
  }
}


static void find_op_in_same_block(Operation *op,
                                  std::vector<Operation *> &group_ops,
                                  std::map<Operation *, int> &op_block_id,
                                  int in_idx) {
  if (std::find(group_ops.begin(), group_ops.end(), op) == group_ops.end()) {
    return;
  }
  if (op_block_id.find(op) != op_block_id.end()) {
    return;
  }
  op_block_id[op] = in_idx;
  for (auto v : op->getOperands()) {
    auto pre_op = v.getDefiningOp();
    if (pre_op == nullptr ||
        isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
      continue;
    }
    if (std::find(group_ops.begin(), group_ops.end(), pre_op) !=
        group_ops.end()) {
      find_op_in_same_block(pre_op, group_ops, op_block_id, in_idx);
    }
  }

  for (auto user : op->getUsers()) {
    if (std::find(group_ops.begin(), group_ops.end(), user) !=
        group_ops.end()) {
      find_op_in_same_block(user, group_ops, op_block_id, in_idx);
    }
  }
}

void show_group(LgInfo *sub_group)
{
  if(sub_group->group_ops.size() == 0)
    return;

  for (auto op : sub_group->group_ops) {
    auto name = module::getName(op).str();
    llvm::errs() << "  op:" << name << "\n";
  }
  for (auto out : sub_group->group_outs) {
    llvm::errs() << "    out:" << module::getName(out).str() << "\n";
  }
  for (auto in : sub_group->group_ins) {
    llvm::errs() << "    in:" << module::getName(in).str() << "\n";
  }
}

int64_t GroupMethod::get_group_cycle(LgInfo *sub_group)
{
  int64_t group_cost = 0;
  shape_secs_t ori_group_shape_secs;

  std::vector<std::pair<Value, int64_t>> value_size;
  if (!init_group_data_secs(*sub_group, ori_group_shape_secs, value_size)){
    return 0;
  }
  BasicTimeStepPtr time_step = std::make_shared<BasicTimeStep>();
  if (!time_step->assignTimeStep(*sub_group, ori_group_shape_secs, true)) {
    return 0;
  }
  if (!update_data_split(time_step, *sub_group, ori_group_shape_secs)) {
    return 0;
  }
  if (!stripe_mine_max_slice(*sub_group, ori_group_shape_secs,
                              time_step->get_tensor_infos())) {
    return 0;
  }
  group_cost = cycle_calculator_->getGroupCycle(time_step, ori_group_shape_secs, sub_group->type);

  return group_cost;
}

Operation* GroupMethod::cut_this_group_is_better(LgInfo *sub_group)
{
  llvm::errs() << ">>>>>> try to cut this group" <<":\n";
  show_group(sub_group);

  std::vector<std::pair<Operation*, int>> cut_op_idx;
  for(int i=0; i<sub_group->group_ops.size(); i++)
  {
    auto cur_op = sub_group->group_ops[i];
    // if(isa<tpu::Conv2DOp, tpu::DeconvOp>(cur_op))
    if(isa<tpu::Conv2DOp>(cur_op))
    {
      cut_op_idx.push_back(std::make_pair(cur_op, i));
    }
  }

  if(cut_op_idx.size()==0) //如果group中没有conv就直接返回
  {
    return nullptr;
  }

  //统计原group的cycle
  int64_t original_group_cost = get_group_cycle(sub_group);
  if(original_group_cost == 0){
    return nullptr;
  }
  llvm::errs() << ">>>>>> original_group_cost:" << original_group_cost <<":\n";
  //尝试所有可以切分的方案,找出耗时最短的方案
  int64_t min_cost = llvm::maxIntN(64);
  Operation* cut_op = nullptr;
  for(int i=0; i<cut_op_idx.size(); i++)
  {
    auto global_op = cut_op_idx[i].first;
    int64_t idx = cut_op_idx[i].second;
    int64_t global_op_cost = cycle_calculator_->getGlobalLayerCycle(global_op);

    int64_t left_sub_group_cost = 0;
    int64_t right_sub_group_cost = 0;
    LgInfo left_sub_group, right_sub_group;
    if(idx-1 >= 0)
    {
      get_layer_group(left_sub_group, sub_group->group_ops, 0, idx-1);
      if(left_sub_group.group_ops.size()==1){
        left_sub_group_cost = cycle_calculator_->getGlobalLayerCycle(left_sub_group.group_ops.back());
      }
      else{
        left_sub_group_cost = get_group_cycle(&left_sub_group);
      }
    }
    if(idx+1 <= sub_group->group_ops.size()-1)
    {
      get_layer_group(right_sub_group, sub_group->group_ops, idx+1, sub_group->group_ops.size()-1);
      if(right_sub_group.group_ops.size()==1){
        right_sub_group_cost = cycle_calculator_->getGlobalLayerCycle(right_sub_group.group_ops.back());
      }
      else{
        right_sub_group_cost = get_group_cycle(&right_sub_group);
      }
    }
    int64_t cut_group_cost = left_sub_group_cost + right_sub_group_cost + global_op_cost;
    if(cut_group_cost < original_group_cost && cut_group_cost < min_cost)
    {
      min_cost = std::min(min_cost, cut_group_cost);
      cut_op = global_op;
    }

    llvm::errs() << ">>>>>> number of attempts:" << i << ":\n";
    llvm::errs() << ">>>>>> cut idx:" << idx << ":\n";
    llvm::errs() << ">>>>>> left group:" << ":\n";
    show_group(&left_sub_group);
    llvm::errs() << ">>>>>> right group:" << ":\n";
    show_group(&right_sub_group);
    llvm::errs() << ">>>>>> global_op_cost: " << global_op_cost << " left_sub_group_cost: " << left_sub_group_cost <<" right_sub_group_cost: "<< right_sub_group_cost <<":\n";
    llvm::errs() << ">>>>>> cut_group_cost:" << cut_group_cost <<":\n";
    llvm::errs() << ">>>>>> original_group_cost:" << original_group_cost <<":\n";
  }

  return cut_op;
}

void GroupMethod::try_cut_some_group(LgPassIR *pass_ir, std::vector<std::vector<Operation *>> &base_groups)
{
  if (module::isDebugCmdEnable("disable_group_cut")) {
    return;
  }
  int grp_num = base_groups.size();
  LgInfo sub_group;
  for (int64_t i = 0; i < grp_num; i++) {
    if (base_groups[i].size() > 1) {
      sub_group.group_id = i;
      sub_group.group_ops.assign(base_groups[i].begin(), base_groups[i].end());
      sub_group.update_group_io(LgPass::OPTIONS.opt);
      set_group_type(sub_group);

      Operation* cut_op = cut_this_group_is_better(&sub_group);
      if(cut_op)
      {
        llvm::errs() << "find cut op!!"<<module::getName(cut_op).str()<<"\n";
        processWhenOpFail(pass_ir, sub_group, base_groups, grp_num, cut_op);
        show_group(&sub_group);
        for (auto it = base_groups[i].begin(); it != base_groups[i].end();) {
            if (std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), *it) == sub_group.group_ops.end()) {
                it = base_groups[i].erase(it);
            } else {
                ++it;
            }
        }
      }
    }
  }
}

Operation* check_single_group_could_be_load(LgInfo &sub_group)
{

  std::vector<std::pair<Operation *, int>> vec_op_hsecs;
  // vec_op_hsecs.push_back(std::make_pair(nullptr, -1));
  get_group_max_secs(sub_group, vec_op_hsecs);
  std::sort(vec_op_hsecs.begin(), vec_op_hsecs.end(),pair_op_int_Sort_by_int);

  shape_secs_t shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;

  // 判断切分后内存是否能加载
  if (!init_group_data_secs(sub_group, shape_secs, value_size)) {
      llvm::errs() << "init_group_data_secs fail\n";
      return vec_op_hsecs[0].first;
    }

  // //考虑反推shape是否正常
  TensorInfo tensor_infos;
  Operation *fail_op = nullptr;
  if (stripe_mine_idx_slice2(sub_group, shape_secs, tensor_infos, fail_op) == false) {
      llvm::errs() << "stripe_mine_idx_slice2 fail, remove fail_op: "
                    << module::getName(fail_op).str() << "\n";
      return fail_op;
    }

  return nullptr;
}

void make_sure_all_group_could_be_load(LgPassIR *pass_ir, std::vector<std::vector<Operation *>> &base_groups)
{

  int grp_num = base_groups.size();
  LgInfo sub_group;
  for (int64_t i = 0; i < grp_num; i++) {
    if (base_groups[i].size() > 1) {
      sub_group.group_id = i;
      sub_group.group_ops.assign(base_groups[i].begin(), base_groups[i].end());
      sub_group.update_group_io(LgPass::OPTIONS.opt);
      set_group_type(sub_group);

      Operation* fail_op = check_single_group_could_be_load(sub_group);
      // if(fail_op && sub_group.group_ops.size() > 3){
      if(fail_op){
        processWhenOpFail(pass_ir, sub_group, base_groups, grp_num, fail_op);

        for (auto it = base_groups[i].begin(); it != base_groups[i].end();) {
            if (std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), *it) == sub_group.group_ops.end()) {
                it = base_groups[i].erase(it);
            } else {
                ++it;
            }
        }

      }
    }
  }
}

// void make_sure_enough_slice_for_multicore(LgPassIR *pass_ir, std::vector<std::vector<Operation *>> &base_groups, int corenum)
// {
//   int grp_num = base_groups.size();
//   LgInfo sub_group;
//   for (int64_t i = 0; i < grp_num; i++) {
//     if (base_groups[i].size() > 1) {
//       sub_group.group_id = i;
//       sub_group.group_ops.assign(base_groups[i].begin(), base_groups[i].end());
//       sub_group.update_group_io(LgPass::OPTIONS.opt);
//       set_group_type(sub_group);

//       std::vector<std::pair<Operation *, int>> vec_op_hsecs;
//       shape_secs_t max_shape_secs = get_group_max_secs(sub_group, vec_op_hsecs);
//       std::sort(vec_op_hsecs.begin(), vec_op_hsecs.end(),pair_op_int_Sort_by_int);

//       llvm::errs()<< "attention!!! sub_group op size:" << sub_group.group_ops.size() << "\n";

//       // 排除输出h为1的op
//       if(max_shape_secs.nsecs*max_shape_secs.hsecs < corenum && vec_op_hsecs[0].second==1)
//       {
//         llvm::errs()<<" op "<<module::getName(vec_op_hsecs[0].first)<<" output h dim is 1, delete it!"<<"\n";
//         processWhenOpFail(pass_ir, sub_group, base_groups, grp_num, vec_op_hsecs[0].first);
//         for (auto it = base_groups[i].begin(); it != base_groups[i].end();) {
//           if (std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), *it) == sub_group.group_ops.end()) {
//               it = base_groups[i].erase(it);
//           } else {
//               ++it;
//           }
//         }
//         vec_op_hsecs.clear();
//         max_shape_secs = get_group_max_secs(sub_group, vec_op_hsecs);
//         std::sort(vec_op_hsecs.begin(), vec_op_hsecs.end(),pair_op_int_Sort_by_int);
//       }

//       std::vector<std::pair<Operation *, int>> vec_op_idx;
//       while(max_shape_secs.nsecs*max_shape_secs.hsecs < 0.5 * corenum){
//         llvm::errs()<< "max_shape_secs n:" << max_shape_secs.nsecs
//         << " c:" << max_shape_secs.csecs << " d:" << max_shape_secs.dsecs
//         << " h:" << max_shape_secs.hsecs << " w:" << max_shape_secs.wsecs
//         <<" corenum "<< corenum << "\n";
//         // 删除h维度最小的算子以及其后面的算子
//         Operation* fail_op = vec_op_hsecs[0].first;
//         llvm::errs() << module::getName(fail_op).str() << "  output h dim " << vec_op_hsecs[0].second <<" too small, delete it from group!\n";

//         auto it = std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), fail_op);
//         for (; it!=sub_group.group_ops.end(); ++it) {
//           int index = std::distance(sub_group.group_ops.begin(), it);
//           vec_op_idx.push_back(std::make_pair(*it, index));
//         }
//         it = std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), fail_op);
//         llvm::errs()<<" grp_idx "<< i << " before erase size: " << sub_group.group_ops.size() << " ins: " <<sub_group.group_ins.size() << " outs: " <<sub_group.group_outs.size() <<"\n";
//         sub_group.group_ops.erase(it, sub_group.group_ops.end());
//         sub_group.update_group_io(LgPass::OPTIONS.opt);
//         set_group_type(sub_group);
//         llvm::errs()<<" grp_idx "<< i << " after erase size: " << sub_group.group_ops.size() << " ins: " <<sub_group.group_ins.size() << " outs: " <<sub_group.group_outs.size() <<"\n";

//         // auto it = std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), fail_op);
//         // int index = std::distance(sub_group.group_ops.begin(), it);
//         // vec_op_idx.push_back(std::make_pair(fail_op, index));
//         // llvm::errs()<< "attention!!! sub_group op size:" << sub_group.group_ops.size() <<" erase idx "<< index << "\n";
//         // llvm::errs()<<" grp_idx "<< i << " before erase size: " << sub_group.group_ops.size() << " ins: " <<sub_group.group_ins.size() << " outs: " <<sub_group.group_outs.size() <<"\n";
//         // sub_group.group_ops.erase(std::remove(sub_group.group_ops.begin(), sub_group.group_ops.end(), fail_op),sub_group.group_ops.end());
//         // sub_group.update_group_io(LgPass::OPTIONS.opt);
//         // set_group_type(sub_group);
//         // llvm::errs()<<" grp_idx "<< i << " after erase size: " << sub_group.group_ops.size() << " ins: " <<sub_group.group_ins.size() << " outs: " <<sub_group.group_outs.size() <<"\n";

//         // processWhenOpFail(pass_ir, sub_group, base_groups, grp_num, fail_op);
//         // for (auto it = base_groups[i].begin(); it != base_groups[i].end();) {
//         //   if (std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), *it) == sub_group.group_ops.end()) {
//         //       it = base_groups[i].erase(it);
//         //   } else {
//         //       ++it;
//         //   }
//         // }

//         if(sub_group.group_ops.size()==0){
//           break;
//         }
//         vec_op_hsecs.clear();
//         max_shape_secs = get_group_max_secs(sub_group, vec_op_hsecs);
//         std::sort(vec_op_hsecs.begin(), vec_op_hsecs.end(),pair_op_int_Sort_by_int);
//       }

//       //h 维度比较小的分成一组
//       if(vec_op_idx.size()>0){
//         std::sort(vec_op_idx.begin(), vec_op_idx.end(), pair_op_int_Sort_by_int);

//         std::vector<Operation*> tmp_group;
//         for(auto op_with_index:vec_op_idx){
//           tmp_group.push_back(op_with_index.first);
//         }

//         // 处理同一个tensor输入两个op的情况
//         for(auto op : sub_group.group_ops)
//         {
//           int count = 0;
//           for (auto user : op->getUsers())
//           {
//             if(std::find(tmp_group.begin(), tmp_group.end(), user)!=tmp_group.end())
//               count++;
//           }

//           if(count>=2)
//           {
//             llvm::errs()<<" op "<<module::getName(op)<<" has multi user in new group"<<"\n";
//             tmp_group.insert(tmp_group.begin(), op);
//             auto it = std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), op);
//             sub_group.group_ops.erase(it);
//           }
//         }

//         base_groups.push_back(tmp_group);
//         pass_ir->ILP_time_steps.push_back(std::vector<ILPTimeStepPtr>());
//         pass_ir->map_l2m_load.push_back(
//         std::map<std::pair<Value, int>, int, value_compare2>());

//         for (auto it = base_groups[i].begin(); it != base_groups[i].end();) {
//           if (std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), *it) == sub_group.group_ops.end()) {
//               it = base_groups[i].erase(it);
//           } else {
//               ++it;
//           }
//         }
//       }
//     }
//   }
// }

void make_sure_enough_slice_for_multicore(LgPassIR *pass_ir, std::vector<std::vector<Operation *>> &base_groups, int corenum)
{
  int grp_num = base_groups.size();
  LgInfo sub_group;
  for (int64_t i = 0; i < grp_num; i++) {
    if (base_groups[i].size() > 1) {
      sub_group.group_id = i;
      sub_group.group_ops.assign(base_groups[i].begin(), base_groups[i].end());
      sub_group.update_group_io(LgPass::OPTIONS.opt);
      set_group_type(sub_group);

      std::vector<std::pair<Operation *, int>> vec_op_hsecs;
      shape_secs_t max_shape_secs = get_group_max_secs(sub_group, vec_op_hsecs);
      std::sort(vec_op_hsecs.begin(), vec_op_hsecs.end(),pair_op_int_Sort_by_int);
      llvm::errs()<< "attention!!! sub_group op size:" << sub_group.group_ops.size() << "\n";

      while(max_shape_secs.nsecs*max_shape_secs.hsecs < corenum){
        llvm::errs()<< "max_shape_secs n:" << max_shape_secs.nsecs
        << " c:" << max_shape_secs.csecs << " d:" << max_shape_secs.dsecs
        << " h:" << max_shape_secs.hsecs << " w:" << max_shape_secs.wsecs
        <<" corenum "<< corenum << "\n";
        // 删除h维度最小的算子
        Operation* fail_op = vec_op_hsecs[0].first;
        llvm::errs() << module::getName(fail_op).str() << "  output h dim " << vec_op_hsecs[0].second <<" too small, delete it from group!\n";

        processWhenOpFail(pass_ir, sub_group, base_groups, grp_num, fail_op);
        for (auto it = base_groups[i].begin(); it != base_groups[i].end();) {
          if (std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), *it) == sub_group.group_ops.end()) {
              it = base_groups[i].erase(it);
          } else {
              ++it;
          }
        }

        if(sub_group.group_ops.size()==0){
          break;
        }
        vec_op_hsecs.clear();
        max_shape_secs = get_group_max_secs(sub_group, vec_op_hsecs);
        // for debug
        if(vec_op_hsecs.size()==0){
          break;
        }
        llvm::errs()<<"vec_op_hsecs.size() : "<<vec_op_hsecs.size()<<"\n";
        for(auto it:vec_op_hsecs)
        {
          llvm::errs() << module::getName(it.first).str() << "\n";
        }
        // for debug
        std::sort(vec_op_hsecs.begin(), vec_op_hsecs.end(),pair_op_int_Sort_by_int);
      }
    }
  }
}

void GroupMethod::l2m_process(LgPassIR *pass_ir, int grp_idx, std::vector<std::pair<Value, int64_t>>& value_size) {
  llvm::errs() << "process l2m...\n";
  auto& grp_time_step = pass_ir->ILP_time_steps[grp_idx];
  auto& map_l2m_load = pass_ir->map_l2m_load[grp_idx];
  int ts_count = grp_time_step[0]->ts_count;
  int core_num_per_pipe0 = grp_time_step[0]->ncdhw_steps.size();
  for (auto itr : grp_time_step[0]->vec_l2m_value_info) {
    llvm::errs() << "check Value:" << module::getName(itr.value).str()
                  << ", slice_idx:" <<itr.slice_idx
                  << ", pipe0 load ts:" << itr.load_ts << "\n";
    int parallel_core_num = core_num_per_pipe0;
    int min = itr.load_ts;
    for (int j = 1; j < grp_time_step.size(); j++) { //遍历除第1个流水外的其他流水，第1个流水最长
      parallel_core_num += grp_time_step[j]->ncdhw_steps.size();
      for (auto itr3 = grp_time_step[j]->vec_l2m_value_info.begin();
                itr3 != grp_time_step[j]->vec_l2m_value_info.end(); ++itr3) {
        if (itr3->value == itr.value && itr3->slice_idx == itr.slice_idx) {
          llvm::errs() << "find in pipe:" <<j<< ", load ts:" << itr3->load_ts << "\n";
          if (itr3->load_ts < min) {
            min = itr3->load_ts;
          }
        }
      }
    }
    if (parallel_core_num > 1) {
      if (map_l2m_load.find(min) == map_l2m_load.end()) {
        map_l2m_load[min] = std::vector<l2m_value_info>();
      }
      map_l2m_load[min].push_back(itr);
    }
  }

  for (int m = -1; m < ts_count; m++) {
    if (map_l2m_load.find(m) != map_l2m_load.end()) {
      for (auto itr: map_l2m_load[m]) {
        llvm::errs() << " Value:" << module::getName(itr.value).str()
                    << " slice_idx:" << itr.slice_idx << " load ts:" << m<< " free ts:" << itr.free_ts << "\n";
      }
    }
  }

  int total_weight_size = 0, l2_mem_size = 128*1024*1024;
  int weight_num = value_size.size();
  for (auto it2: value_size) {
    total_weight_size += it2.second;
  }
  l2mem_alloc_Ptr l2mem_alloc_ptr = std::make_shared<l2mem_alloc>();
  std::vector<Value> value_l2m;
  if (total_weight_size > l2_mem_size) {
    int share_mem_size = 0;
    for (int i = weight_num - 1; i > 0; i--) {
      std::vector<std::pair<Value, int64_t>> value_size_l2m;
      std::vector<int64_t> value_l2m_addr;
      value_l2m.clear();
      share_mem_size += value_size[i].second;
      total_weight_size = 0;
      int addr = 0;
      for (auto it2: value_size) {
        total_weight_size += it2.second;
        if (total_weight_size > l2_mem_size - (int)(share_mem_size*1.5)) {
          break;
        }
        value_size_l2m.push_back(it2);
        value_l2m.push_back(it2.first);
        value_l2m_addr.push_back(addr);
        addr +=it2.second;
      }
      l2mem_alloc_ptr->clear();
      for (auto it3: value_size_l2m) {
        auto name = module::getName(it3.first).str();
        l2mem_alloc_ptr->alloc(-1, name, it3.first, it3.second);
      }

      auto& map_l2m_load = pass_ir->map_l2m_load[grp_idx];
      std::map<int, std::vector<l2m_value_info>> map_l2m_free;
      bool failed = false;
      for (int m = -1; m < ts_count; m++) {
        //处理在该时隙需要释放的l2m tensor
        if (map_l2m_free.find(m) != map_l2m_free.end()) {
          for (auto it3:map_l2m_free[m]) {
            auto name = module::getName(it3.value).str();
            l2mem_alloc_ptr->free(it3.slice_idx, name);
          }
        }
        //处理在该时隙需要分配的l2m tensor
        if (map_l2m_load.find(m) != map_l2m_load.end()) {
          for (auto it3:map_l2m_load[m]) {
            if (std::find(value_l2m.begin(), value_l2m.end(), it3.value) == value_l2m.end()) {
              auto name = module::getName(it3.value).str();
              failed = l2mem_alloc_ptr->alloc(it3.slice_idx, name, it3.value, it3.size);
              if (failed) {
                break;
              }
              //记录当前分配的l2m tensor待释放的时隙
              if (map_l2m_free.find(it3.free_ts) == map_l2m_free.end()) {
                map_l2m_free[it3.free_ts] = std::vector<l2m_value_info>();
              }
              map_l2m_free[it3.free_ts].push_back(it3);
            }
          }
        }
        if (failed) {
          break;
        }
      }
    }
  } else {
    llvm::errs() << "l2m enough \n";
    for (auto it3: value_size) {
      value_l2m.push_back(it3.first);
      auto name = module::getName(it3.first).str();
      l2mem_alloc_ptr->alloc(-1, name, it3.first, it3.second);
    }
  }

  for (int m = -1; m < ts_count; m++) {
    if (map_l2m_load.find(m) != map_l2m_load.end()) {
      for (auto& itr: map_l2m_load[m]) {
        if (itr.slice_idx > 0 && std::find(value_l2m.begin(), value_l2m.end(), itr.value) != value_l2m.end()) {
          llvm::errs() << "value:" << module::getName(itr.value).str() << ",set valid false\n";
          itr.valid = false;
        }
      }
    }
  }
  pass_ir->lg_l2mem_alloc_ptr.push_back(l2mem_alloc_ptr);
}

bool GroupMethod::is_same_pipeline(LgPassIR *pass_ir, int core_id, int grp_idx, int& vec_ncdhw_idx,
                                  TensorInfo& tensor_infos, LgInfo &sub_group,
                                  std::vector<std::vector<int64_t>> vec_ncdhw, std::vector<int>& sec_per_cores) {
  bool all_slice_same = false;
  for (int n = 0; n < pass_ir->ILP_time_steps[grp_idx].size(); n++) { // 遍历历史流水线
    std::vector<std::vector<int64_t>> &ncdhw_steps = pass_ir->ILP_time_steps[grp_idx][n]->ncdhw_steps.begin()->second;
    if (ncdhw_steps.size() == sec_per_cores[core_id]) { // 历史流水线与当前有相同slice数量
      all_slice_same = true;
      for (int m = 0; m < sec_per_cores[core_id]; m++) {
        std::vector<int64_t>& his_steps = ncdhw_steps[m];
        std::vector<int64_t> ncdhw = vec_ncdhw[vec_ncdhw_idx + m];
        slice_info_t &slice_info = tensor_infos[sub_group.group_ops[0]->getOperand(0)] .slice_info; // todo 这里是使用于单分支
        // for (auto itr = tensor_infos.begin(); itr != tensor_infos.end(); ++itr) {
        if (slice_info.n[his_steps[0]].second != slice_info.n[ncdhw[0]].second ||
            slice_info.c[his_steps[1]].second != slice_info.c[ncdhw[1]].second ||
            slice_info.d[his_steps[2]].second != slice_info.d[ncdhw[2]].second ||
            slice_info.h[his_steps[3]].second != slice_info.h[ncdhw[3]].second ||
            slice_info.w[his_steps[4]].second != slice_info.w[ncdhw[4]].second) {
            all_slice_same = false;
            break;
        }
      }
      if (all_slice_same) {
        llvm::errs() << "core " << core_id << ",all slice shape same with pipeline " << n << ", skip ILP\n";
        for (int m = 0; m < sec_per_cores[core_id]; m++) {
          std::vector<int64_t> ncdhw = vec_ncdhw[vec_ncdhw_idx + m];
          pass_ir->ILP_time_steps[grp_idx][n]->addSliceNcdhwSteps(core_id, ncdhw);
        }
        vec_ncdhw_idx += sec_per_cores[core_id];
        break;
      }
    }
  }
  return all_slice_same?true:false;
}

Operation* GroupMethod::ilp_for_single_group(LgPassIR *pass_ir, LgInfo &sub_group, int grp_idx, int core_num, bool l2m_switch, bool train){
  std::vector<std::pair<Operation *, int>> vec_op_hsecs;
  vec_op_hsecs.push_back(std::make_pair(nullptr, -1));
  std::sort(vec_op_hsecs.begin(), vec_op_hsecs.end(), pair_op_int_Sort_by_int);

  shape_secs_t shape_secs;
  std::vector<std::pair<Value, int64_t>> value_size;
  init_group_data_secs(sub_group, shape_secs, value_size);
  llvm::errs() << "init shape_secs, n:" << shape_secs.nsecs
                << " c:" << shape_secs.csecs << " d:" << shape_secs.dsecs
                << " h:" << shape_secs.hsecs << " w:" << shape_secs.wsecs <<'\n';

  if(core_num > 1){
    std::vector<std::pair<Operation *, int>> vec_op_hsecs_tmp;
    shape_secs_t max_shape_secs = get_group_max_secs(sub_group, vec_op_hsecs_tmp);
    llvm::errs() << "max_shape_secs, n:" << max_shape_secs.nsecs
              << " c:" << max_shape_secs.csecs << " d:" << max_shape_secs.dsecs
              << " h:" << max_shape_secs.hsecs << " w:" << max_shape_secs.wsecs <<'\n';
    int64_t secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
    int64_t dhw_secs = shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;

    int new_secs = (secs + core_num - 1)/core_num*core_num;
    int sz = new_secs - secs;
    if (sz > 0) {
      llvm::errs() <<"algin secs:"<<secs<<" to "<<new_secs<<"\n";

      std::vector<int> vec_secs;
      std::vector<int> vec_dhw_secs;
      std::vector<shape_secs_t> vec_shape_secs;
      for (int m = 0; m < sz; m++) {
        // update_shape_secs2(sub_group, shape_secs, dhw_secs, max_shape_secs);
        update_shape_secs_for_ilp_group(shape_secs, max_shape_secs);

        llvm::errs() << "update shape shape_secs, n:" << shape_secs.nsecs
        << " c:" << shape_secs.csecs << " d:" << shape_secs.dsecs
        << " h:" << shape_secs.hsecs << " w:" << shape_secs.wsecs <<'\n';

        int tmp_secs = shape_secs.nsecs *shape_secs.csecs *shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
        vec_secs.push_back(std::abs(tmp_secs - new_secs));
        vec_shape_secs.push_back(shape_secs);
        vec_dhw_secs.push_back(dhw_secs);
        if (tmp_secs > new_secs) {
            break;
          }
      }
    if (vec_secs.size() > 0) {
        auto closest_secs = *std::min_element(vec_secs.begin(), vec_secs.end()); //找到相差最小的
        auto it2 = std::find(vec_secs.begin(), vec_secs.end(), closest_secs);
        int pos = std::distance(vec_secs.begin(), it2);
        shape_secs = vec_shape_secs[pos];
        dhw_secs = vec_dhw_secs[pos];
        llvm::errs() << "new shape_secs, n:" << shape_secs.nsecs
              << " c:" << shape_secs.csecs << " d:" << shape_secs.dsecs
              << " h:" << shape_secs.hsecs << " w:" << shape_secs.wsecs <<'\n';
      }
    }
  }

  std::sort(value_size.begin(), value_size.end(), Sort_by_int);
  int try_count = 0, max_try_count = 10;
  TensorInfo tensor_infos;
  Operation *fail_op = sub_group.group_ops[0];

  bool ret = false;
  while (true) {
    if (++try_count > max_try_count) {
      llvm::errs() <<"layer group fail\n";
      return fail_op;
    }
    int64_t secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
    bool l2m_en = l2m_switch && secs > 1 && core_num > 1;

    llvm::errs() << "shape_secs, n:" << shape_secs.nsecs
                  << " c:" << shape_secs.csecs << " d:" << shape_secs.dsecs
                  << " h:" << shape_secs.hsecs << " w:" << shape_secs.wsecs
                  << " try_count:" << try_count<< " l2m_en:" << l2m_en << "\n";

    ret = stripe_mine_idx_slice2(sub_group, shape_secs, tensor_infos,fail_op);
    if(!ret){
      llvm::errs() << module::getName(fail_op).str() << " stripe_mine_idx_slice2 fail"<<"\n";
      return fail_op;
    }
    update_tensor_infos(sub_group, tensor_infos);
    sub_group.update_bank_info();

    int vec_ncdhw_idx = 0;
    std::vector<std::vector<int64_t>> vec_ncdhw;
    auto sec_per_cores = get_sec_per_cores(shape_secs, vec_ncdhw, core_num, tensor_infos);
    ILPTimeStepPtr ilp_timeStep;
    pass_ir->ILP_time_steps[grp_idx].clear();

    for (int core_id = 0; core_id < core_num; core_id++) {
      if (sec_per_cores[core_id] == 0) {
        break;
      }

      if (is_same_pipeline(pass_ir, core_id, grp_idx, vec_ncdhw_idx, tensor_infos, sub_group, vec_ncdhw, sec_per_cores)) {
        continue;
      }

      int slice_idx = 0;
      std::vector<op_var_pos_info> op_var_bound = createOverlapStrategy(sub_group, sec_per_cores[core_id]);
      std::vector<std::pair<Value, int64_t>> tmp_value_size;
      if (sec_per_cores[core_id] >
          0) { // 始终将小的权重提前加进来,放在lmem的最后区域，减少后续数据依赖，减少时隙间同步
        tmp_value_size.assign(value_size.begin(), value_size.end());
      }
      // 预先去掉最大的20%的权重，以便backward_gen_ilp_var2中可提前更多时隙加载它们
      for (int i = 0; i < tmp_value_size.size() * 0.2;
            i++) { // 0.2是经验值，不一定恰当
        tmp_value_size.pop_back();
      }

      int64_t load_bytes_for_next_ts = 0;
      ilp_timeStep = std::make_shared<ILPTimeStep>(sub_group, sec_per_cores[core_id]);
      while(sec_per_cores[core_id]-- > 0) {
        std::vector<int64_t> ncdhw = vec_ncdhw[vec_ncdhw_idx++];
        ilp_timeStep->addSliceNcdhwSteps(core_id, ncdhw);
        llvm::errs() << "slice process, n:" << ncdhw[0]
                      << " c:" << ncdhw[1] << " d:" << ncdhw[2]
                      << " h:" << ncdhw[3] << " w:" << ncdhw[4]<< " ncdhw_idx:" << vec_ncdhw_idx - 1 << "\n";
        ret = backward_gen_ilp_var2(
            sub_group, shape_secs, tensor_infos, cycle_calculator_,
            *ilp_timeStep, ncdhw, slice_idx, op_var_bound,
            pass_ir->returnOp, load_bytes_for_next_ts,
            tmp_value_size, fail_op, l2m_en, sec_per_cores[core_id] == 0, 4);
        if(!ret){
          llvm::errs() <<"backward_gen_ilp_var2 fail" <<" core_id "<<core_id<<"\n";
          return fail_op;
        }
        slice_idx++;
      }

      ret = ilp_timeStep->merge_small_cycle_op(tensor_infos);
      if (!ret) {
        llvm::errs() << "ilp_timeStep->merge_small_cycle_op fail\n";
        return fail_op;
      }

      ret = ilp_timeStep->prepare(tensor_infos);
      if (!ret) {
        llvm::errs() << "ilp_timeStep->prepare fail\n";
        return fail_op;
      }

      ret = ilp_timeStep->run();
      if (!ret) {
        llvm::errs() << "ilp_timeStep->run fail\n";
        return fail_op;
      }

      mem_alloc_status alloc_status;
      ret = ilp_timeStep->mem_alloc(alloc_status, tmp_value_size, tensor_infos);
      if (!ret) {
        printf("ilp_timeStep->mem_alloc fail\n");
        return fail_op;
      }
      llvm::errs() << "grp" << grp_idx << ",core" << core_id << ", mem_alloc success\n";
      pass_ir->ILP_time_steps[grp_idx].emplace_back(ilp_timeStep);
    }

    if(ret){
      llvm::errs() << "ilp_timeStep success\n";
      if (l2m_en) {
        l2m_process(pass_ir, grp_idx, value_size);
      }
      break;
    }
  }

  pass_ir->shape_secs.emplace_back(shape_secs);
  pass_ir->lg_tensor_infos_.push_back(tensor_infos);
  pass_ir->lg_infos.push_back(sub_group);

  return nullptr;
}

void GroupMethod::init_ilp_base_groups(LgPassIR* pass_ir, LgInfo& sub_group, std::vector<std::vector<Operation *>> &base_groups){

  int grp_num = pass_ir->tmp_base_groups.size();
  for (int64_t i = 0; i < grp_num; i++) {
    if (pass_ir->tmp_base_groups[i].size() > 1) {
      sub_group.group_ops.assign(pass_ir->tmp_base_groups[i].begin(), pass_ir->tmp_base_groups[i].end());
      sub_group.update_group_io(LgPass::OPTIONS.opt);
      int in_idx = 0;
      std::map<Operation *, int> op_block_id;
      for (auto in : sub_group.group_ins) {
        for (auto user : in.getUsers()) {
          find_op_in_same_block(user, sub_group.group_ops, op_block_id, in_idx);
        }
        in_idx++;
      }

      bool first_group = true;
      bool have_valid_grp = false;
      for (int j = 0; j < in_idx; j++) {
        std::vector<Operation *> block_ops;
        for (auto itr = op_block_id.begin(); itr != op_block_id.end(); ++itr) {
          if (j == itr->second) {
            block_ops.push_back(itr->first);
          }
        }
        if (block_ops.size() > 1) {
          have_valid_grp = true;
          if (first_group) {
            pass_ir->tmp_base_groups[i].assign(block_ops.begin(),
                                               block_ops.end());
            first_group = false;
          } else {
            llvm::errs() << "add new block_ops\n";
            pass_ir->tmp_base_groups.push_back(block_ops);
          }
        }
      }
      if (!have_valid_grp) {
        llvm::errs() << "not have_valid_grp\n";
        pass_ir->tmp_base_groups[i].clear();
      }
    }
  }

  if (pass_ir->branch_parallel) {
    get_base_branch_groups(base_groups, pass_ir->subnet_ops,
                           pass_ir->subnet_return_opds);
  } else {
    get_base_dfs_topo_groups(base_groups, pass_ir->subnet_ops,
                             pass_ir->tmp_base_groups);
  }

  grp_num = base_groups.size();
  for (int64_t i = grp_num - 1; i >= 0; --i) {
    if (base_groups[i].size() > 1) {
      pass_ir->ILP_time_steps.push_back(std::vector<ILPTimeStepPtr>());
      pass_ir->map_l2m_load.push_back(
          std::map<int, std::vector<l2m_value_info>>());
    }
  }

}

void GroupMethod::ilp_layer_group(LgPassIR *pass_ir) {
  llvm::errs() << "\n"
               << "=======================================================\n"
               << "*********** ilp_layer_group **********\n"
               << "=======================================================\n";
  //------------------------part0: pre processing----------------------------------------------------
  auto start = std::chrono::high_resolution_clock::now();
  int core_num = 1;
  if (dyn_cast<MultiCoreInterface>(BM168x::instance())) {
   core_num = module::getCoreNum();
  }
  LgInfo sub_group;
  std::vector<std::vector<Operation *>> base_groups;
  init_ilp_base_groups(pass_ir, sub_group, base_groups);

  make_sure_all_group_could_be_load(pass_ir, base_groups);
  try_cut_some_group(pass_ir, base_groups);
  if(core_num > 1){
    make_sure_enough_slice_for_multicore(pass_ir, base_groups, core_num);
  }

  //------------------------part1: processing----------------------------------------------------
  int grp_num = base_groups.size();
  // module::setCoreNum(core_num);
  // if (use_all_hard_core_to_compile) {
  //   core_num = cycle_calculator_->getMultiCoreNum();
  // } else {
  //   core_num = module::getCoreNum();
  // }

  bool l2m_switch = false;
  if (module::isDebugCmdEnable("enable_l2m")) {
    l2m_switch = true;
  }
  bool train = true;
  int grp_idx = 0;

  std::cout<<"init grp_num is "<<grp_num<<std::endl;
  for (int64_t i = 0; i < grp_num; i++) {
    if (base_groups[i].size() > 1) {
      sub_group.group_id = i;
      sub_group.group_ops.assign(base_groups[i].begin(), base_groups[i].end());
      sub_group.update_group_io(LgPass::OPTIONS.opt);
      set_group_type(sub_group);
      llvm::errs() << ">>>>>> process group" << grp_idx << ":\n";
      show_group(&sub_group);
      sub_group.dump_lginfo();

      // deal single group until sucess or break all op into global op
      while(true){
        Operation* fail_op = ilp_for_single_group(pass_ir, sub_group, grp_idx, core_num, l2m_switch, train);
        if(fail_op){
          llvm::errs() <<"some op fail!!!" << "\n";
          processWhenOpFail(pass_ir, sub_group, base_groups, grp_num, fail_op);
          for (auto it = base_groups[i].begin(); it != base_groups[i].end();) {
              if (std::find(sub_group.group_ops.begin(), sub_group.group_ops.end(), *it) == sub_group.group_ops.end()) {
                  it = base_groups[i].erase(it);
              } else {
                  ++it;
              }
          }
          if(sub_group.group_ops.size()==1){
            break;
          }
        }else{
          grp_idx++;
          break;
        }
      }
    }
  }

  //------------------------part2: post processing----------------------------------------------------
  std::shared_ptr<dot_graph> dot_graph_log_ok = std::make_shared<dot_graph>();
  std::vector<Operation *> subnet_ops;
  subnet_ops.assign(pass_ir->subnet_ops.begin(), pass_ir->subnet_ops.end());
  createSubnetGraph(subnet_ops, dot_graph_log_ok);

  grp_idx = 0;
  for (auto lg_info : pass_ir->lg_infos) {
    for (auto op : lg_info.group_ops) {
      auto name = module::getName(op).str();
      dot_graph_log_ok->add_node_label(name + "_ori",
                                       "grp_" + std::to_string(grp_idx));
    }
    grp_idx++;
  }
  dot_graph_log_ok->export_dot("backward_gen_ilp_var_all_ok");

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  llvm::errs() << "get ilp_layer_group time:" << elapsed.count() << "\n";
}

void GroupMethod::process(LgPassIR *pass_ir) {
  std::vector<LgInfo> &lg_infos = pass_ir->lg_infos;
  SetVector<Operation *> &subnet_ops = pass_ir->subnet_ops;

  runmode_ = getRunMode(subnet_ops[0]);
  switch (LgPass::OPTIONS.opt) {
  case 1:
    simple_layer_group(lg_infos, subnet_ops);
    break;
  case 2:
    dynamic_programming_layer_group_with_cluster(lg_infos, subnet_ops);
    break;
  case 3:
    ilp_layer_group(pass_ir);
    break;
  default:
    simple_layer_group(lg_infos, subnet_ops);
    break;
  }
}

void GroupMethod::get_final_groups(
    std::vector<LgInfo> &lg_infos,
    const std::vector<std::vector<Operation *>> &base_groups) {
  int64_t start_idx, end_idx;
  LgInfo lg_info;
  for (size_t i = 0; i < base_groups.size(); ++i) {
    start_idx = 0;
    auto &base_group = base_groups[i];
    auto &cut_result = cut_results_[i];
    for (size_t j = 0; j < cut_result.size(); ++j) {
      end_idx = cut_result[j];
      get_layer_group(lg_info, base_group, start_idx, end_idx);
      if (lg_info.group_ops.size() > 1 ||
          false == LgPass::OPTIONS.group_by_cores) {
        lg_infos.push_back(lg_info);
      }
      LLVM_DEBUG({ lg_info.dump_lginfo(); });
      start_idx = end_idx + 1;
    }
  }
}

void GroupMethod::show_cut_results() {
  LLVM_DEBUG(for (size_t i = 0; i < cut_results_.size(); ++i) {
    auto &cut_result = cut_results_[i];
    llvm::errs() << "base group[" << i << "] cut results: ";
    for (size_t j = 0; j < cut_result.size(); ++j) {
      llvm::errs() << cut_result[j] << ", ";
    }
    llvm::errs() << "\n";
  });
}

/// The pass of layer group searching
class LayerGroupSearchPass : public LgPass {
public:
  LayerGroupSearchPass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    auto group_method = GroupMethod(options_.opt);
    group_method.process(pass_ir);
    return true;
  }
  virtual std::string name() override { return "LayerGroupSearchPass"; }
  virtual std::string brief() override {
    return "Searching the optimal layer groups";
  }

private:
  LgOptions options_;
};

std::unique_ptr<LgPass> CreateLayerGroupSearchPass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new LayerGroupSearchPass(options));
}

} // namespace tpu
} // namespace tpu_mlir
