# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import shutil
import os
import subprocess
import logging
import utils.pattern_counter


def _os_system_log(cmd_str):
    # use subprocess to redirect the output stream
    # the file for saving the output stream should be set if using this function
    logging.info("[Running]: %s", cmd_str)

    process = subprocess.Popen(cmd_str,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline().strip()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output)

    process.wait()
    ret = process.returncode

    if ret == 0:
        logging.info("[Success]: %s", cmd_str)
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))


def _os_system(cmd: list, save_log: bool = False, mute: bool = False):
    cmd_str = " ".join(cmd)
    if mute:
        ret = subprocess.call(cmd_str,
                              shell=True,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        assert ret == 0
        return
    if not save_log:
        print("[Running]: {}".format(cmd_str))
        ret = os.system(cmd_str)
        if ret == 0:
            print("[Success]: {}".format(cmd_str))
        else:
            raise RuntimeError("[!Error]: {}".format(cmd_str))
    else:
        _os_system_log(cmd_str)


def get_matched_patterns(log_file: str = ""):
    if log_file:
        matcher = utils.pattern_counter.PatternCounter(log_file)
        matcher.count_matched_patterns()
        return matcher.success_counter
    return {}


def mlir_opt_for_top(mlirfile: str,
                     opt_mlirfile: str,
                     add_postprocess: str = "",
                     count_patterns: bool = False):
    cmd = ["tpuc-opt", mlirfile, "--shape-infer"]
    if len(add_postprocess) > 0:
        cmd.extend([f"--add-postprocess=\"type={add_postprocess}\""])
#    cmd.extend(["--canonicalize", "--extra-optimize", "-o", opt_mlirfile])
    cmd.extend(["--canonicalize", "--extra-optimize", "--deinit", "--dev-placement", "-o", opt_mlirfile]) # ymha: dev-placement
    log_file = ""
    if count_patterns:
        log_file = "top_patterns.log"
        cmd.extend(["--debug", "> {} 2>&1".format(log_file)])
    _os_system(cmd)
    return get_matched_patterns(log_file)


def mlir_lowering(top_mlir: str,
                  tpu_mlir: str,
                  mode: str,
                  chip: str,
                  num_device: int = 1,
                  num_core: int = 1,
                  cali_table: str = None,
                  asymmetric: bool = False,
                  quantize_table: str = None,
                  customization_format: str = None,
                  fuse_preprocess: bool = False,
                  aligned_input: bool = False,
                  ignore_f16_overflow: bool = False,
                  do_winograd: bool = False,
                  q_group_size: int = 0,
                  count_patterns: bool = False,
                  addr_mode: str = "auto",
                  mute: bool = False):
    cmd = [
        "tpuc-opt", top_mlir,
        "--processor-assign=\"chip={} num_device={} num_core={} addr_mode={}\"".format(
            chip.lower(), num_device, num_core, addr_mode)
    ]
    mode = mode.upper()
    # asymmetric = False  # TODO: always using symmetric, as asymmetric not good
    if cali_table != None:
        cali_param = "--import-calibration-table=\"file={} asymmetric={}\"".format(
            cali_table, asymmetric)
        cmd.extend([cali_param])
    #do extra conversion for differnet chips
    cmd.extend(["--processor-top-optimize"])
    if fuse_preprocess:
        fuse_pre_param = "--fuse-preprocess=\"mode={} customization_format={} align={}\"".format(
            mode, customization_format, aligned_input)
        cmd.extend([fuse_pre_param])
    qtable = ""
    if quantize_table:
        assert (tpu_mlir.endswith(".mlir"))
        weight_name = tpu_mlir[:-len(".mlir")] + "_qtable_weights.npz"
        qtable = "qtable={} weightFileName={}".format(quantize_table, weight_name)
    lower_param = "--convert-top-to-tpu=\"mode={} {} asymmetric={} doWinograd={} ignore_f16_overflow={} q_group_size={}\"".format(
        mode, qtable, asymmetric, do_winograd, ignore_f16_overflow, q_group_size)
    cmd.extend([
        lower_param,
        "--canonicalize",
        "--weight-fold",
        "-o",
        tpu_mlir,
    ])
    log_file = ""
    if count_patterns:
        log_file = "lowering_patterns.log"
        cmd.extend(["--debug", "> {} 2>&1".format(log_file)])
    _os_system(cmd, mute=mute)
    return get_matched_patterns(log_file)


def mlir_to_model(tpu_mlir: str,
                  model: str,
                  final_mlir: str,
                  dynamic: bool = False,
                  quant_input: bool = False,
                  quant_output: bool = False,
                  quant_input_list: str = "",
                  quant_output_list: str = "",
                  disable_layer_group: bool = False,
                  opt: int = 2,
                  merge_weight: bool = False,
                  op_divide: bool = False,
                  embed_debug_info: bool = False,
                  group_by_cores: str = "auto",
                  model_version: str = "",
                  count_patterns: bool = False,
                  compress_mode: str = "none",
                  debug_cmd: str = ""):
    # generate final mlir
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={} quant_input_list={} quant_output_list={}"'.format(
        quant_input, quant_output, quant_input_list, quant_output_list)
    lg_param = ''
    if not disable_layer_group:
        lg_param = '--layer-group="opt={} group_by_cores={} compress_mode={}"'.format(
            opt, group_by_cores, compress_mode)
    subnet_param = '--subnet-divide="dynamic={}"'.format(dynamic)
    address_assign_param = '--address-assign'
    if merge_weight:
        address_assign_param = '--address-assign="merge_weight=true weight_map_file=_weight_map.csv"'
    distribute_param = f"--dev-parallel"
    parallel_param = f"--core-parallel"

    op_divide_param = ""
    if op_divide:
        op_divide_param = "--op-divide"
    # yapf: disable
    cmd = [
        "tpuc-opt",
        tpu_mlir,
        "--mlir-disable-threading",
        strip_io_quant_param,
        "--processor-tpu-optimize",
    ]
    # yapf: enable

    if embed_debug_info:
        # save the optimized tpu.mlir
        tpu_opt_mlir = final_mlir[:-10] + "tpu_opt.mlir"
        cmd.extend([
            "-o",
            tpu_opt_mlir,
            debug_cmd
        ])
        _os_system(cmd)
        cmd = [
            "tpuc-opt",
            tpu_opt_mlir
        ]

    cmd.extend([
        distribute_param,
        "--weight-reorder",
        op_divide_param,
        subnet_param,
        "--op-reorder",
        lg_param,
        parallel_param,
        address_assign_param,
        "-o",
        final_mlir,
        debug_cmd
    ])
    log_file = ""
    if count_patterns:
        log_file = "tpu_patterns.log"
        cmd.extend(["--debug", "> {} 2>&1".format(log_file)])
    _os_system(cmd)

    # codegen based on final mlir
    codegen_param = (
        f'--codegen="model_file={model} embed_debug_info={str(embed_debug_info).capitalize()} model_version={str(model_version).lower()}"'
    )
    cmd = [
        "tpuc-opt",
        final_mlir,
        codegen_param,
        "-o /dev/null",
    ]
    _os_system(cmd)

    out_dir = model.rsplit(".", maxsplit=1)[0]
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(final_mlir, os.path.join(out_dir, 'final.mlir'))
    try:
        if model.endswith(".bmodel") and not dynamic:
            # The suffix of the profile file is not consistent.
            # bm1684 uses ".dat", bm1684x uses ".txt".
            _os_system(["mv compiler_profile_0.[td][xa]t", model + ".compiler_profile_0.txt"])
            _os_system(["mv net_0.profile", model + ".net_0.profile"])
    except RuntimeError:
        pass

    return get_matched_patterns(log_file)


def origin_mlir_txt_to_bmodel(
    converter,
    model_name: str,
    mode: str,
    chip: str,
    add_postprocess: str = "",
    num_device: int = 1,
    num_core: int = 1,
    cali_table: str = None,
    asymmetric: bool = False,
    quantize_table: str = None,
    customization_format: str = None,
    fuse_preprocess: bool = False,
    aligned_input: bool = False,
    ignore_f16_overflow: bool = False,
    do_winograd: bool = False,
    q_group_size: int = 0,
    dynamic: bool = False,
    quant_input: bool = False,
    quant_output: bool = False,
    quant_input_list: str = "",
    quant_output_list: str = "",
    disable_layer_group: bool = False,
    opt: int = 2,
    merge_weight: bool = False,
    op_divide: bool = False,
    embed_debug_info: bool = False,
    addr_mode: str = "auto",
    group_by_cores: str = "auto",
    model_version: str = "",
    count_patterns: bool = False,
    compress_mode: str = "none"
):
    bmodel = f"{model_name}_{mode}.bmodel"

    options = [
        "--init",
        "--shape-infer",
    ]
    if len(add_postprocess) > 0:
        options.extend([f'--add-postprocess="type={add_postprocess}"'])
    options.extend(["--canonicalize", "--extra-optimize"])

    options.extend(
        [
            f'--processor-assign="chip={chip.lower()} num_device={num_device} num_core={num_core} addr_mode={addr_mode}"'
        ]
    )
    mode = mode.upper()
    # asymmetric = False  # TODO: always using symmetric, as asymmetric not good
    if cali_table != None:
        cali_param = '--import-calibration-table="file={} asymmetric={}"'.format(
            cali_table, asymmetric
        )
        options.extend([cali_param])
    # do extra conversion for differnet chips
    options.extend(["--processor-top-optimize"])
    if fuse_preprocess:
        fuse_pre_param = ('--fuse-preprocess="mode={} customization_format={} align={}"'.format(
            mode, customization_format, aligned_input))
        options.extend([fuse_pre_param])
        fuse_pre_param = (
            '--fuse-preprocess="mode={} customization_format={} align={}"'.format(
                mode, customization_format, aligned_input
            )
        )
        options.extend([fuse_pre_param])

    lower_param = '--convert-top-to-tpu="mode={} asymmetric={} doWinograd={} ignore_f16_overflow={} q_group_size={}"'.format(
        mode, asymmetric, do_winograd, ignore_f16_overflow, q_group_size
    )
    options.extend(
        [
            lower_param,
            "--canonicalize",
            "--weight-fold",
        ]
    )
    # generate final mlir
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={} quant_input_list={} quant_output_list={}"'.format(
        quant_input, quant_output, quant_input_list, quant_output_list)
    lg_param = ""
    if not disable_layer_group:
        lg_param = '--layer-group="opt={} group_by_cores={} compress_mode={}"'.format(
            opt, group_by_cores, compress_mode)
    subnet_param = '--subnet-divide="dynamic={}"'.format(dynamic)
    address_assign_param = '--address-assign'
    if merge_weight:
        address_assign_param = (
            '--address-assign="merge_weight=true weight_map_file=_weight_map.csv"')
    distribute_param = f"--dev-parallel"
    parallel_param = f"--core-parallel"

    op_divide_param = ""
    if op_divide:
        op_divide_param = "--op-divide"
    # codegen based on final mlir
    codegen_param = f'--codegen="model_file={bmodel} embed_debug_info={str(embed_debug_info).capitalize()} model_version={str(model_version).lower()} bmodel_only=True"'

    options.extend(
        [
            strip_io_quant_param,
            "--processor-tpu-optimize",
            distribute_param,
            "--weight-reorder",
            op_divide_param,
            subnet_param,
            "--op-reorder",
            lg_param,
            parallel_param,
            address_assign_param,
            codegen_param,
            f'--deinit="no_save_weight=True"'
        ]
    )
    log_file = ""
    if count_patterns:
        log_file = "tpu_patterns.log"
        options.extend(["--debug"])

    import pymlir
    mlir_txt = converter.get_mlir_txt()
    print("origin_mlir: ")
    print(mlir_txt)
    print("options: ", options)
    pymlir.run_pass_pipeline(mlir_txt, options)

    return get_matched_patterns(log_file)


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True, fuzzy_match=False):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    if fuzzy_match:
        cmd.append('--fuzzy_match')
    _os_system(cmd)


# TOPTOTOSA
def top_to_tosa(top_mlir: str, tosa_mlir: str, includeWeight: bool = False):
    cmd = ["tpuc-opt", top_mlir]
    lower_param = "--convert-top-to-tosa=\"includeWeight="
    if includeWeight:
        lower_param += "True\""
    else:
        lower_param += "False\""
    cmd.extend([lower_param, "--canonicalize", "-o", tosa_mlir])
    _os_system(cmd)


# TOSATOObj
def tosa_to_llvm(tosa_mlir: str, objfile: str):
    cmd = ["mlir-opt", tosa_mlir]
    lower_param = (
        "--pass-pipeline=\"builtin.module("
        "func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-arith, tosa-to-tensor, tosa-to-scf), "
        "convert-tensor-to-linalg, "
        "func.func(canonicalize, linalg-bufferize, convert-linalg-to-affine-loops, affine-loop-fusion, affine-simplify-structures, lower-affine), "
        "func-bufferize, "
        "func.func(tensor-bufferize, llvm-request-c-wrappers), "
        "arith-expand, arith-bufferize, normalize-memrefs, convert-scf-to-cf, "
        "convert-math-to-llvm, convert-arith-to-llvm, convert-func-to-llvm, convert-cf-to-llvm, "
        "convert-bufferization-to-memref, memref-expand, expand-strided-metadata, finalize-memref-to-llvm, "
        "canonicalize, llvm-legalize-for-export, reconcile-unrealized-casts)\""
        "| mlir-translate --mlir-to-llvmir "
        "| llc -mtriple=x86_64-unknown-linux-gnu --filetype=obj")
    cmd.extend([lower_param, "-o", objfile])
    _os_system(cmd)


# Model inference on CPU
def model_inference_cpu(objfile: str, output_size: str):
    # generate executable file: a.out
    print("Generating executable file a.out ...")
    ccompiler = "clang"
    cfile = "/workspace/tpu-mlir/capi/runtime_cpu.c"
    model = objfile
    lib1 = "/workspace/tpu-mlir/capi/lib/libmlir_c_runner_utils.so.17git"
    lib2 = "/workspace/tpu-mlir/capi/lib/libmlir_runner_utils.so.17git"
    lib3 = "/workspace/tpu-mlir/capi/lib/libmlir_float16_utils.so.17git"
    lib4 = "-lm"
    cflag = "-fPIC"
    cmd = [ccompiler, cfile, model, lib1, lib2, lib3, lib4, cflag]
    _os_system(cmd)
    print("Successfully generate executable file a.out!")
    # execute model inference
    print("Runing ...")
    cmd1 = ["./a.out", output_size]
    _os_system(cmd1)
    print("Inference ends successfully! Results are saved in inference_result.txt.")


# Extra tool: delete file in current directory
def delete_file(file: str):
    cmd = ["rm -f", file]
    _os_system(cmd)
