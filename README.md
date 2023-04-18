# Language Grasp-Anything

Language Grasp-Anything is an open source project that combines speech recognition, target detection and robotic 6-dof grasping to generate feasible grasping poses of specific objects from cluttered scenes. It builds on the Whisper speech recognition model, the GroundingDINO detection model, and the Graspness grasping model.

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Whisper:

```python
pip install -U openai-whisper
```

See the [whisper official page](https://github.com/openai/whisper#setup) if you have other questions for the installation.

Install Grounding DINO:

```python
python -m pip install -e GroundingDINO
```

Install Graspness:

For Graspness, please refer [here](https://github.com/rhett-chen/graspness_implementation).

## Getting Started

- Download the checkpoint for Grounding Dino:

  ```python
  cd Grounded-Segment-Anything
  
  wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  ```

- Download the checkpoint for Graspness:

  ```python
  https://drive.google.com/file/d/1HXhO6z2XNAnGW4BiGHBy83cVa-d32AeV/view?usp=share_link
  ```

- Run demo:

  ```python
  export CUDA_VISIBLE_DEVICES=0
  python real_grasp.py --infer --vis
  ```

## Acknowledgments

This project is based on the following repositories:

- [Whisper](https://github.com/openai/whisper#setup)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grapness](https://github.com/rhett-chen/graspness_implementation)
