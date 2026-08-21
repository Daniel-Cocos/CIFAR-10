{
  description = "CIFAR-10 CNN";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        cudaSupport = true;
        allowUnfree = true;
      };
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python312
        python312Packages.torch # PyTorch core
        python312Packages.torchvision # datasets + transforms
        python312Packages.numpy
        python312Packages.matplotlib # plot images & curves
        python312Packages.seaborn # confusion matrix
        python312Packages.pillow # image handling
        python312Packages.tensorboard # training viz
      ];
    };
  };
}
