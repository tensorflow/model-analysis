"""TensorFlow Model Analysis Polymer Dependencies"""

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library_external")

def tensorflow_model_analysis_polymer_workspace():
    """Download TensorFlow Model Analysis polymer dependencies."""

    web_library_external(
        name = "org_googlewebcomponents_google_apis",
        licenses = ["notice"],  # BSD-3-Clause
        sha256 = "1e0a83f1af1978875789620edd837e6a06c1316f3bf6c2ed14d8450a7d4d3251",
        urls = [
            "https://mirror.bazel.build/github.com/GoogleWebComponents/google-apis/archive/v1.1.7.tar.gz",
            "https://github.com/GoogleWebComponents/google-apis/archive/v1.1.7.tar.gz",
        ],
        strip_prefix = "google-apis-1.1.7",
        path = "/google-apis",
        srcs = [
            "google-apis.html",
            "google-client-loader.html",
            "google-js-api.html",
            "google-legacy-loader.html",
            "google-maps-api.html",
            "google-plusone-api.html",
            "google-youtube-api.html",
        ],
        deps = [
            "@org_polymer",
            "@org_polymerelements_iron_jsonp_library",
        ],
    )

    web_library_external(
        name = "org_googlewebcomponents_google_chart",
        licenses = ["notice"],  # BSD-3-Clause
        sha256 = "e4a959deb8ad9660ea4ee5552e87e1e064f4b76008bf0fe37b4f4ce51817d480",
        urls = [
            "https://mirror.bazel.build/github.com/GoogleWebComponents/google-chart/archive/v1.1.1.tar.gz",
            "https://github.com/GoogleWebComponents/google-chart/archive/v1.1.1.tar.gz",
        ],
        strip_prefix = "google-chart-1.1.1",
        path = "/google-chart",
        srcs = [
            "charts-loader.html",
            "google-chart.css",
            "google-chart.html",
            "google-chart-loader.html",
        ],
        deps = [
            "@org_googlewebcomponents_google_apis",
            "@org_polymer",
            "@org_polymerelements_iron_ajax",
            "@org_polymerlabs_promise_polyfill",
        ],
    )

    web_library_external(
        name = "org_polymer_iron_pages",
        licenses = ["notice"],  # BSD-3-Clause
        sha256 = "9a1b8e6b2d1dd11f94d7aa674c811a1e6b7dd766678e3650228deb109520612a",
        urls = [
            "https://mirror.bazel.build/github.com/PolymerElements/iron-pages/archive/v1.0.9.tar.gz",
            "https://github.com/PolymerElements/iron-pages/archive/v1.0.9.tar.gz",
        ],
        strip_prefix = "iron-pages-1.0.9",
        path = "/iron-pages",
        srcs = ["iron-pages.html"],
        deps = [
            "@org_polymer",
            "@org_polymer_iron_resizable_behavior",
            "@org_polymer_iron_selector",
        ],
    )

    web_library_external(
        name = "org_polymerelements_iron_ajax",
        licenses = ["notice"],  # BSD-3-Clause
        sha256 = "4979d6e5601deeede65f35ec4c89416ea2fbebc78c113ff9305c6687acd9ddcc",
        urls = [
            "https://mirror.bazel.build/github.com/PolymerElements/iron-ajax/archive/v1.4.1.tar.gz",
            "https://github.com/PolymerElements/iron-ajax/archive/v1.4.1.tar.gz",
        ],
        strip_prefix = "iron-ajax-1.4.1",
        path = "/iron-ajax",
        srcs = [
            "iron-ajax.html",
            "iron-request.html",
        ],
        deps = [
            "@org_polymer",
            "@org_polymerlabs_promise_polyfill",
        ],
    )

    web_library_external(
        name = "org_polymerelements_iron_jsonp_library",
        licenses = ["notice"],  # BSD-3-Clause
        sha256 = "15667734d1be5c1ec108e9753b6c66a0c44ee32da312ff1b483856eb0bd861a4",
        urls = [
            "https://mirror.bazel.build/github.com/PolymerElements/iron-jsonp-library/archive/v1.0.5.tar.gz",
            "https://github.com/PolymerElements/iron-jsonp-library/archive/v1.0.5.tar.gz",
        ],
        strip_prefix = "iron-jsonp-library-1.0.5",
        path = "/iron-jsonp-library",
        srcs = [
            "iron-jsonp-library.html",
        ],
        deps = [
            "@org_polymer",
        ],
    )

    web_library_external(
        name = "org_polymerlabs_promise_polyfill",
        licenses = ["notice"],  # BSD-3-Clause
        sha256 = "d83edb667c393efb3e7b40a2c22d439e1d84056be5d36174be6507a45f709daa",
        urls = [
            "https://mirror.bazel.build/github.com/PolymerLabs/promise-polyfill/archive/v1.0.1.tar.gz",
            "https://github.com/PolymerLabs/promise-polyfill/archive/v1.0.1.tar.gz",
        ],
        strip_prefix = "promise-polyfill-1.0.1",
        path = "/promise-polyfill",
        srcs = [
            "Gruntfile.js",
            "Promise-Statics.js",
            "Promise.js",
            "Promise.min.js",
            "promise-polyfill-lite.html",
            "promise-polyfill.html",
        ],
        deps = [
            "@org_polymer",
        ],
    )
