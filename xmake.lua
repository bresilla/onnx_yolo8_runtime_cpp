add_rules("mode.debug", "mode.release")

add_requires("opencv", {system = true})
add_requires("onnxruntime")
-- add_requires("spdlog")

target("run")
    -- set_languages("cxx17")
    set_kind("binary")
    add_files("src/main.cpp")
    add_files("src/inference.cpp")
    add_packages("opencv")
    add_includedirs("include")
