/* -----------------------------------------------------------------------------
 * Copyright (C) 2025, Electronics and Telecommunications Research Institute (ETRI)
 * All rights reserved.
 *
 * @Author: Youngmok Ha
 *
 * This code includes an implementation of "DevicePlacementPass".
 *
 * "DevicePlacementPass" operates when an argument "dev-placement" is set for tpuc-opt
 *  ex) tpuc-opt xxxx_origin.mlir --shape-infer --canonicalize --extra-optimize --deinit --dev-placement -o xxxx.mlir
 *
 * -----------------------------------------------------------------------------
 */


#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/json.hpp"

#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "device_placement"

#include <cstdlib>
#include <regex>

using namespace llvm;
using json = nlohmann::ordered_json;

static std::string extractTypeStr(mlir::Type type) {
    std::string str;

    llvm::raw_string_ostream os(str);
    type.print(os);

    return os.str();
}

static std::string extractLocStr(mlir::Location loc) {
    std::regex pattern(R"(loc\(\"([^\"]+)\"\))");
    std::string str;
    std::smatch match;

    llvm::raw_string_ostream os(str);
    loc.print(os);

    if(std::regex_search(os.str(), match, pattern)){
        return match[1];
    }

    return "unknown";
}

namespace tpu_mlir {
namespace top {

extern void devicePlacementThroughputDynamicProgramming(json* dp_input);
extern void devicePlacementLatencyGreedy(json* dp_input);

class DevicePlacementPass : public DevicePlacementBase<DevicePlacementPass> {
public:
  DevicePlacementPass() {}
  void runOnOperation() override {

    /* 
       TODO: make func calls to
       - get memory_size_per_fpga
       - get num_cpus 
    */
    double memory_size_per_fpga = 629145600;
    int num_devs = (int) module::getDeviceNum();
    int num_cpus = 1;

    std::unordered_map<std::string, int> loc2id_map;
    std::unordered_map<std::string, int64_t> typeByteSize = {
        {"f32", 4}, {"f64", 8}, {"f16", 2}, {"bf16", 2} };

    auto mOp = getOperation();

    /*
        the structure of dp_input:
        - maxSizePerFPGA (floating-point)
        - maxFPGAs (integer)
        - maxCPUs (integer)
        - nodes (array)
        - edges (array)

        the structure of each node:
        - id (integer)
        - supportedOnFpga (boolean)
        - cpuLatency  (floating-point)
        - fpgaLatency (floating-point)
        - isBackwardNode (boolean)
        - colorClass (integer, optional)
        - size (floating-point)  

        the structure of each edge:
        - sourceId (integer)
        - destId (integer)
        - cost (floating-point)

        *refer to https://github.com/msr-fiddle/dnn-partitioning
    */
    json dp_input;

    std::vector<json> nodes, edges;

    int cnt = 0;
    for (auto func : mOp.getOps<FuncOp>()) {
      // iterate over operation nodes
      func.walk([&](Operation *op) {
        int id = cnt;
        double memory_size = 0.0;

        json node;

        std::string name = op->getName().getStringRef().str();
        std::string loc = extractLocStr(op->getLoc());
        if (loc == "unknown"){
            loc += "." + name;
        }

        loc2id_map[loc] = id;

        for (unsigned i = 0; i < op->getNumResults(); ++i) {
            int64_t op_memory_size = 0;
            if(!(op->getResult(i).getType().isa<NoneType>())){
                int64_t n, c, h, w;
                std::string tensor_type = extractTypeStr(module::getElementType(op->getResult(i)));
                module::getNCHW(op->getResult(i), n, c, h, w, false);
                op_memory_size = n * c * h * w * typeByteSize[tensor_type];
            }
            memory_size += (double) op_memory_size;
        }

        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            Operation *oprd = op->getOperand(i).getDefiningOp();

            if(oprd == nullptr){
                continue;
            }

            json edge;

            auto oprd_name = oprd->getName().getStringRef().str();
            auto oprd_loc = extractLocStr(oprd->getLoc());
            if(loc2id_map.find(oprd_loc) == loc2id_map.end()){
                oprd_loc += "." + oprd_name;
            }
            auto oprd_id = loc2id_map[oprd_loc];
            
            int64_t oprd_memory_size = 0;
            if(!(op->getOperand(i).getType().isa<NoneType>())){
                std::string tensor_type = extractTypeStr(module::getElementType(op->getOperand(i)));
                int64_t n, c, h, w;
                module::getNCHW(op->getOperand(i), n, c, h, w, false);
                oprd_memory_size = n * c * h * w * typeByteSize[tensor_type];
            }
            memory_size += oprd_memory_size;

            edge["sourceId"] = oprd_id;
            edge["destId"] = id;
            edge["cost"] = 0.0;
            edges.push_back(edge);
        }

        node["name"] = name;
        node["id"] = id;
        node["supportedOnFpga"] = false;
        node["cpuLatency"] = 0.0;
        node["fpgaLatency"] = 0.0;
        node["isBackwardNode"] = false;
        node["size"] = memory_size;

        nodes.push_back(node); 
        cnt++;
      });
    }

    dp_input["maxSizePerFPGA"] = memory_size_per_fpga;
    dp_input["maxFPGAs"] = num_devs;
    dp_input["maxCPUs"] = num_cpus;
    dp_input["nodes"] = nodes;
    dp_input["edges"] = edges;
    
    devicePlacementThroughputDynamicProgramming(&dp_input);
//    devicePlacementLatencyGreedy(&dp_input);

  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDevicePlacementPass() {
  return std::make_unique<DevicePlacementPass>();
}
} // namespace top
} // namespace tpu_mlir
