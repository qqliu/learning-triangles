licenses(["notice"])

cc_library(
    name = "StronglyConnectedComponents",
    hdrs = ["StronglyConnectedComponents.h"],
    deps = [
        "//gbbs",
        "//gbbs/helpers:resizable_table",
        "//gbbs/helpers:sparse_table",
        "//gbbs/semiasym:graph_filter",
    ],
)

cc_binary(
    name = "StronglyConnectedComponents_main",
    srcs = ["StronglyConnectedComponents.cc"],
    deps = [":StronglyConnectedComponents"],
)

package(
    default_visibility = ["//visibility:public"],
)
