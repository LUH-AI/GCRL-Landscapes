{
  description = "Virtual Environment for Python";
  inputs.systems.url = "github:nix-systems/default";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    systems,
  }:
    flake-utils.lib.eachSystem (import systems)
    (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      lib = pkgs.lib;
    in {
      packages = flake-utils.lib.flattenTree {
      };

      devShells.default = let
        pythonPackages = pkgs.python310Packages;
        makeNixLDWrapper = program: (pkgs.runCommand "${program.pname}-nix-ld-wrapped" { } ''
          mkdir -p $out/bin
          for file in ${program}/bin/*; do
            new_file=$out/bin/$(basename $file)
            echo "#! ${pkgs.bash}/bin/bash -e" >> $new_file
            echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NIX_LD_LIBRARY_PATH"' >> $new_file
            echo 'exec -a "$0" '$file' "$@"' >> $new_file
            chmod +x $new_file
          done
        '');
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
        ];
        NIX_LD = pkgs.runCommand "ld.so" {} ''
          ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
        '';
        # commented out for now as it does make problems with the normal system. may fix problems with packages
        LD_LIBRARY_PATH = NIX_LD_LIBRARY_PATH;
        QT_PLUGIN_PATH = "${pkgs.qt6.qtbase}/${pkgs.qt6.qtbase.qtPluginPrefix}:${pkgs.qt6.qtwayland}/${pkgs.qt6.qtbase.qtPluginPrefix}";
        # lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
        buildInputs = (with pkgs; [
          glxinfo
          hatch
          openssl
          swig
          zlib
          redis
        ]) ++ ([
          pythonPackages.venvShellHook
          (makeNixLDWrapper pythonPackages.python)
          # (makeNixLDWrapper pkgs.uv)
          pkgs.uv
        ]);
        nativeBuildInputs = (with pkgs; [
            ruff
            pre-commit
            gcc11
          ]);
      };
    });
}
