# flake.nix, run with `nix develop`
# Run with `nix-shell cuda-fhs.nix`
{ pkgs ? import <nixpkgs> {} }:
let
   # Change according to the driver used: stable, beta
   nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.stable;
in
(pkgs.buildFHSEnv {
  name = "cuda-env";
  targetPkgs = pkgs: with pkgs; [ 
    git
    gitRepo
    gnupg
    autoconf
    curl
    procps
    gnumake
    util-linux
    m4
    gperf
    unzip
    cudatoolkit
    nvidiaPackage
    libGLU libGL
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
    ncurses5
    stdenv.cc
    binutils
    python310Packages.python
  ];
  multiPkgs = pkgs: with pkgs; [ zlib openssl cacert ];
  extraMounts = [
    {
      source = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
      target = "/etc/ssl/certs/ca-bundle.crt";
      recursive = false;
    }
  ];
  extraEnv = {
    SSL_CERT_FILE = "/etc/ssl/certs/ca-bundle.crt";
    REQUESTS_CA_BUNDLE = "/etc/ssl/certs/ca-bundle.crt";
  };
  NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
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
    # pkgs.cudaPackages.cudatoolkit
    # pkgs.cudaPackages.cuda_nvcc
  ];
  NIX_LD = pkgs.runCommand "ld.so" {} ''
    ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
  '';

  runScript = "zsh";
  
  profile = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    # export LD_LIBRARY_PATH=${nvidiaPackage}/lib
    export EXTRA_LDFLAGS="-L/lib -L${nvidiaPackage}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
    source .venv/bin/activate
  '';
}).env
