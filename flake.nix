{
  description = "Flake-based development shell for conversational-clustering";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
    in
    {
      devShells = nixpkgs.lib.genAttrs systems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          pythonVersion = pkgs.python311;
          runtimeLibs = [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.libglvnd
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudaPackages.cuda_nvrtc
            pkgs.cudaPackages.libcublas
            pkgs.cudaPackages.cudnn
          ];
          corePackages = [
            pythonVersion
            pkgs.uv
            pkgs.nodejs
            pkgs.pnpm
            pkgs.bun
            pkgs.bashInteractive
            pkgs.zsh
            pkgs.git
          ] ++ runtimeLibs;
          libraryPath = pkgs.lib.makeLibraryPath runtimeLibs;
          shellInit = ''
            export UV_PYTHON="${pythonVersion}/bin/python"
            export UV_PYTHON_DOWNLOADS="never"
            export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
            if [ "$(uname -s)" = "Linux" ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:${libraryPath}:$LD_LIBRARY_PATH"
            else
              export LD_LIBRARY_PATH="${libraryPath}:$LD_LIBRARY_PATH"
            fi

            if [ -f "pyproject.toml" ]; then
              echo "Syncing uv project..."
              uv sync --dev
            fi

            if [ -f "demo_frontend/package.json" ]; then
              echo "Installing demo frontend dependencies..."
              ( cd demo_frontend && pnpm install )
            fi

            if [ -f ".venv/bin/activate" ]; then
              . .venv/bin/activate
            fi
          '';
        in
        {
          default =
            if pkgs.stdenv.isLinux then
              (pkgs.buildFHSEnv {
                name = "conversational-clustering-fhs";
                targetPkgs = _: corePackages;
                runScript = "zsh";
                profile = shellInit;
              }).env
            else
              pkgs.mkShell {
                packages = corePackages;
                shellHook = shellInit;
              };
        }
      );
    };
}
