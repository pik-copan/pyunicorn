
# Documentation: https://devenv.sh/

{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [ zlib git ruff pandoc ];

  cachix.enable = true;

  languages.python = {
    enable = true;
    version = "3.14";
    venv.enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # For interactive use, create an alias `_ruff` pointing to the `ruff` version
  # installed directly by `devenv`, thus avoiding the dynamically linked `ruff`
  # version installed by the `uv` version installed by `devenv`, which can be
  # problematic for Nix-based host systems.
  scripts._ruff.exec = ''
    $DEVENV_PROFILE/bin/ruff "$@"
  '';
}
