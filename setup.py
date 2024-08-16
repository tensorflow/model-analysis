# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sets up TFMA package for Jupyter notebook integration.

The widget is based on the template generated from jupyter-widget's
widget-cookiecutter.
"""
from distutils import log
from distutils import spawn
import os
import platform
import subprocess
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist

# pylint: disable=g-bad-import-order
# It is recommended to import setuptools prior to importing distutils to avoid
# using legacy behavior from distutils.
# https://setuptools.readthedocs.io/en/latest/history.html#v48-0-0
# pylint: enable=g-bad-import-order


# Find the Protocol Compiler.
if "PROTOC" in os.environ and os.path.exists(os.environ["PROTOC"]):
    protoc = os.environ["PROTOC"]
elif os.path.exists("../src/protoc"):
    protoc = "../src/protoc"
elif os.path.exists("../src/protoc.exe"):
    protoc = "../src/protoc.exe"
elif os.path.exists("../vsprojects/Debug/protoc.exe"):
    protoc = "../vsprojects/Debug/protoc.exe"
elif os.path.exists("../vsprojects/Release/protoc.exe"):
    protoc = "../vsprojects/Release/protoc.exe"
else:
    protoc = spawn.find_executable("protoc")

# Get version from version module.
with open("tensorflow_model_analysis/version.py") as fp:
    globals_dict = {}
    exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict["VERSION"]

here = os.path.dirname(os.path.abspath(__file__))
node_root = os.path.join(here, "tensorflow_model_analysis", "notebook", "jupyter", "js")
is_repo = os.path.exists(os.path.join(here, ".git"))

npm_path = os.pathsep.join(
    [
        os.path.join(node_root, "node_modules", ".bin"),
        os.environ.get("PATH", os.defpath),
    ]
)

# Set this to true if ipywidgets js should be built. This would require nodejs.
build_js = os.environ.get("BUILD_JS") is not None

log.set_verbosity(log.DEBUG)
log.info("setup.py entered")
log.info("$PATH=%s" % os.environ["PATH"])


def generate_proto(source, require=True):
    """Invokes the Protocol Compiler to generate a _pb2.py."""

    # Does nothing if the output already exists and is newer than
    # the input.

    if not require and not os.path.exists(source):
        return

    output = source.replace(".proto", "_pb2.py").replace("../src/", "")

    if not os.path.exists(output) or (
        os.path.exists(source) and os.path.getmtime(source) > os.path.getmtime(output)
    ):
        print("Generating %s..." % output)

        if not os.path.exists(source):
            sys.stderr.write("Can't find required file: %s\n" % source)
            sys.exit(-1)

        if protoc is None:
            sys.stderr.write(
                "protoc is not installed nor found in ../src.  Please compile it "
                "or install the binary package.\n"
            )
            sys.exit(-1)

        protoc_command = [protoc, "-I../src", "-I.", "--python_out=.", source]
        if subprocess.call(protoc_command) != 0:
            sys.exit(-1)


def generate_tfma_protos():
    """Generate necessary .proto file if it doesn't exist."""
    generate_proto("tensorflow_model_analysis/proto/config.proto", False)
    generate_proto("tensorflow_model_analysis/proto/metrics_for_slice.proto", False)
    generate_proto("tensorflow_model_analysis/proto/validation_result.proto", False)


class build_py(_build_py):  # pylint: disable=invalid-name
    """Build necessary dependencies."""

    def run(self):
        generate_tfma_protos()
        # _build_py is an old-style class, so super() doesn't work.
        _build_py.run(self)


class develop(_develop):  # pylint: disable=invalid-name
    """Build necessary dependencies in develop mode."""

    def run(self):
        generate_tfma_protos()
        _develop.run(self)


def js_prerelease(command, strict=False):
    """Decorator for building minified js/css prior to another command."""

    class DecoratedCommand(command):
        """Decorated command."""

        def run(self):
            jsdeps = self.distribution.get_command_obj("jsdeps")
            if not is_repo and all(os.path.exists(t) for t in jsdeps.targets):
                # sdist, nothing to do
                command.run(self)
                return

            try:
                self.distribution.run_command("jsdeps")
            except Exception as e:  # pylint: disable=broad-except
                missing = [t for t in jsdeps.targets if not os.path.exists(t)]
                if strict or missing:
                    log.warn("rebuilding js and css failed")
                    if missing:
                        log.error("missing files: %s" % missing)
                    raise e
                else:
                    log.warn("rebuilding js and css failed (not a problem)")
                    log.warn(str(e))
            command.run(self)
            update_package_data(self.distribution)

    return DecoratedCommand


def update_package_data(distribution):
    """update package_data to catch changes during setup."""
    build_py_cmd = distribution.get_command_obj("build_py")
    # distribution.package_data = find_package_data()
    # re-init build_py options which load package_data
    build_py_cmd.finalize_options()


class NPM(Command):
    """NPM builder.

    Builds the js and css using npm.
    """

    description = "install package.json dependencies using npm"

    user_options = []

    node_modules = os.path.join(node_root, "node_modules")

    targets = [
        os.path.join(here, "tensorflow_model_analysis", "static", "extension.js"),
        os.path.join(here, "tensorflow_model_analysis", "static", "index.js"),
        os.path.join(here, "tensorflow_model_analysis", "static", "vulcanized_tfma.js"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def get_npm_name(self):
        npm_name = "npm"
        if platform.system() == "Windows":
            npm_name = "npm.cmd"

        return npm_name

    def has_npm(self):
        npm_name = self.get_npm_name()
        try:
            subprocess.check_call([npm_name, "--version"])
            return True
        except:  # pylint: disable=bare-except
            return False

    def should_run_npm_install(self):
        return self.has_npm()

    def run(self):
        if not build_js:
            return

        has_npm = self.has_npm()
        if not has_npm:
            log.error(
                "`npm` unavailable.  If you're running this command using sudo, make"
                " sure `npm` is available to sudo"
            )

        env = os.environ.copy()
        env["PATH"] = npm_path

        if self.should_run_npm_install():
            log.info(
                "Installing build dependencies with npm.  This may take a while..."
            )
            npm_name = self.get_npm_name()
            subprocess.check_call(
                [npm_name, "install"],
                cwd=node_root,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            os.utime(self.node_modules, None)

        for t in self.targets:
            if not os.path.exists(t):
                msg = "Missing file: %s" % t
                if not has_npm:
                    msg += (
                        "\nnpm is required to build a development version of a widget"
                        " extension"
                    )
                raise ValueError(msg)

        # update package data in case this created new files
        update_package_data(self.distribution)


def _make_extra_packages_tfjs():
    # Packages needed for tfjs.
    return [
        "tensorflowjs>=4.5.0,<5",
    ]


def _make_extra_packages_test():
    # Packages needed for tests
    return [
        "pytest>=8.0",
    ]


def _make_extra_packages_all():
    # All optional packages
    return [
        *_make_extra_packages_tfjs(),
        *_make_extra_packages_test(),
    ]


def select_constraint(default, nightly=None, git_master=None):
    """Select dependency constraint based on TFX_DEPENDENCY_SELECTOR env var."""
    selector = os.environ.get("TFX_DEPENDENCY_SELECTOR")
    if selector == "UNCONSTRAINED":
        return ""
    elif selector == "NIGHTLY" and nightly is not None:
        return nightly
    elif selector == "GIT_MASTER" and git_master is not None:
        return git_master
    else:
        return default


# Get the long description from the README file.
with open("README.md") as fp:
    _LONG_DESCRIPTION = fp.read()

setup_args = {
    "name": "tensorflow_model_analysis",
    "version": __version__,
    "description": "A library for analyzing TensorFlow models",
    "long_description": _LONG_DESCRIPTION,
    "long_description_content_type": "text/markdown",
    "include_package_data": True,
    "data_files": [
        (
            "share/jupyter/nbextensions/tensorflow_model_analysis",
            [
                "tensorflow_model_analysis/static/extension.js",
                "tensorflow_model_analysis/static/index.js",
                "tensorflow_model_analysis/static/index.js.map",
                "tensorflow_model_analysis/static/vulcanized_tfma.js",
            ],
        ),
    ],
    # Make sure to sync the versions of common dependencies (numpy, six, and
    # protobuf) with TF.
    "install_requires": [
        # Sort alphabetically
        "absl-py>=0.9,<2.0.0",
        'apache-beam[gcp]>=2.53,<3;python_version>="3.11"',
        'apache-beam[gcp]>=2.47,<3;python_version<"3.11"',
        "ipython>=7,<8",
        "ipywidgets>=7,<8",
        "numpy>=1.23.5",
        "pandas>=1.0,<2",
        "pillow>=9.4.0",
        'protobuf>=4.25.2,<5;python_version>="3.11"',
        'protobuf>=3.20.3,<5;python_version<"3.11"',
        "pyarrow>=10,<11",
        "rouge-score>=0.1.2,<2",
        "sacrebleu>=2.3,<4",
        "scipy>=1.4.1,<2",
        "six>=1.12,<2",
        "tensorflow"
        + select_constraint(
            default=">=2.15,<2.16",
            nightly=">=2.16.0.dev",
            git_master="@git+https://github.com/tensorflow/tensorflow@master",
        ),
        "tensorflow-estimator>=2.10",
        "tensorflow-metadata"
        + select_constraint(
            default=">=1.15.0,<1.16.0",
            nightly=">=1.16.0.dev",
            git_master="@git+https://github.com/tensorflow/metadata@master",
        ),
        "tfx-bsl"
        + select_constraint(
            default=">=1.15.1,<1.16.0",
            nightly=">=1.16.0.dev",
            git_master="@git+https://github.com/tensorflow/tfx-bsl@master",
        ),
    ],
    "extras_require": {
        "all": _make_extra_packages_all(),
    },
    "python_requires": ">=3.9,<4",
    "packages": find_packages(),
    "zip_safe": False,
    "cmdclass": {
        "build_py": js_prerelease(build_py),
        "develop": js_prerelease(develop),
        "egg_info": js_prerelease(egg_info),
        "sdist": js_prerelease(sdist, strict=True),
        "jsdeps": NPM,
    },
    "author": "Google LLC",
    "author_email": "tensorflow-extended-dev@googlegroups.com",
    "license": "Apache 2.0",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "namespace_packages": [],
    "requires": [],
    "keywords": "tensorflow model analysis tfx",
    "url": "https://www.tensorflow.org/tfx/model_analysis/get_started",
    "download_url": "https://github.com/tensorflow/model-analysis/tags",
}

setup(**setup_args)
