{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.python3Packages.virtualenv
    pkgs.python311Packages.notebook
    pkgs.python311Packages.jupyterlab
    pkgs.python311Packages.pip
    pkgs.python311Packages.opencv4
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.tqdm
    pkgs.python311Packages.pandas
    pkgs.python311Packages.torch
    pkgs.python311Packages.scikit-learn
    pkgs.python311Packages.scikit-image
  ];

  shellHook = ''
    if [ ! -d "venv" ]; then
       virtualenv venv
    fi
    source venv/bin/activate
    pip install -r scripts/requirements_su-et-al.txt
  '';
}
