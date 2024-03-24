set_languages("cxx17")
add_rules("mode.debug", "mode.release")

-- add_includedirs("/doc/work/data/RIWO/onnx_yolo8_runtime/ext/onnxruntime/include")

-- add_includedirs("/doc/work/data/RIWO/onnx_yolo8_runtime/ext/onnxruntime/include")
-- add_linkdirs("/doc/work/data/RIWO/onnx_yolo8_runtime/ext/onnxruntime/lib")
-- add_links("onnxruntime")

add_requires("opencv", {system = true})

-- add_requires("onnxruntime", {configs = {gpu = true}})
add_requires("onnxruntime")
add_requires("spdlog")

target("run")
    -- set_languages("cxx17")
    set_kind("binary")
    add_includedirs("include")
    add_files("src/main.cpp")
    add_files("src/inference.cpp")
    add_packages("opencv")
    add_packages("spdlog")
    add_packages("onnxruntime")
