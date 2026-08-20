{
  description = "CIFAR-10 CNN";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python311
          python311Packages.torch       # PyTorch core
          python311Packages.torchvision # datasets + transforms
          python311Packages.numpy
          python311Packages.matplotlib  # plot images & curves
          python311Packages.seaborn     # confusion matrix
          python311Packages.pillow      # image handling
          python311Packages.tensorboard # training viz
        ];
      };
    };
}
