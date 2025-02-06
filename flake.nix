{
  description = "Virtual Environment for Python";
  inputs.systems.url = "github:nix-systems/default";
  inputs.libfolding.url = "github:CrunchyFlakes/libfolding_flake_nix";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    systems,
    libfolding,
  }:
    flake-utils.lib.eachSystem (import systems)
    (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      lib = pkgs.lib;
      libfolding_pkg = libfolding.packages.${system}.libfolding;
    in {
      packages = flake-utils.lib.flattenTree {
        inherit (pkgs) hello;
      };

      devShells.default = let
        pythonPackages = pkgs.python312Packages;
      in pkgs.mkShell rec {
        venvDir = "./.venv";
        NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.stdenv.cc.cc
          pkgs.stdenv.cc
          pkgs.qt6.qtbase
          pkgs.qt6.qtsvg
          pkgs.qt6.qtdeclarative
          pkgs.qt6.qtwayland
          pkgs.libcxx
          pkgs.openssl
          pkgs.zlib
          libfolding_pkg
        ];
        NIX_LD = pkgs.runCommand "ld.so" {} ''
          ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
        '';
        LD_LIBRARY_PATH = NIX_LD_LIBRARY_PATH;
        QT_PLUGIN_PATH = "${pkgs.qt6.qtbase}/${pkgs.qt6.qtbase.qtPluginPrefix}:${pkgs.qt6.qtwayland}/${pkgs.qt6.qtbase.qtPluginPrefix}";
        # lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
        buildInputs = (with pkgs; [
          pythonPackages.python
          pythonPackages.venvShellHook
          pythonPackages.mujoco
          pythonPackages.black
          glxinfo
          hatch
          stdenv.cc
          stdenv.cc.cc.lib
          openssl
          swig
          zlib
        ]) ++ (with pythonPackages; [
          python
          venvShellHook
          tkinter
        ]) ++ [ libfolding_pkg ];
        nativeBuildInputs = (with pkgs; [
            ruff
            pre-commit
          ]);
      };
    });
}
