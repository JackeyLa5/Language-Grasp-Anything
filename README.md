# Language Grasp-Anything

Language Grasp-Anything is an open source project that combines speech recognition, target detection and robotic 6-dof grasping to generate feasible grasping poses of specific objects from cluttered scenes. It builds on the Whisper speech recognition model, the GroundingDINO detection model, and the Graspness grasping model.

**🚀 A short video demonstration**

<details class="details-reset border rounded-2" open="" style="box-sizing: border-box; display: block; border: var(--borderWidth-thin, 1px) solid var(--color-border-default)  !important; border-radius: var(--borderRadius-medium, 6px)  !important; margin-top: 0px; margin-bottom: 16px; color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary class="px-3 py-2" style="box-sizing: border-box; display: list-item; cursor: pointer; padding-top: var(--base-size-8, 8px)  !important; padding-bottom: var(--base-size-8, 8px)  !important; padding-right: var(--base-size-16, 16px)  !important; padding-left: var(--base-size-16, 16px)  !important; list-style: none; transition: color 80ms cubic-bezier(0.33, 1, 0.68, 1) 0s, background-color, box-shadow, border-color;"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-device-camera-video"><path d="M16 3.75v8.5a.75.75 0 0 1-1.136.643L11 10.575v.675A1.75 1.75 0 0 1 9.25 13h-7.5A1.75 1.75 0 0 1 0 11.25v-6.5C0 3.784.784 3 1.75 3h7.5c.966 0 1.75.784 1.75 1.75v.675l3.864-2.318A.75.75 0 0 1 16 3.75Zm-6.5 1a.25.25 0 0 0-.25-.25h-7.5a.25.25 0 0 0-.25.25v6.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-6.5ZM11 8.825l3.5 2.1v-5.85l-3.5 2.1Z"></path></svg><span>&nbsp;</span><span aria-label="Video description chatbot.mp4" class="m-1" style="box-sizing: border-box; margin: var(--base-size-4, 4px)  !important;">chatbot.mp4</span><span>&nbsp;</span><span class="dropdown-caret" style="box-sizing: border-box; border-bottom-color: rgba(0, 0, 0, 0); border-left-color: rgba(0, 0, 0, 0); border-right-color: rgba(0, 0, 0, 0); border-style: solid; border-width: var(--borderWidth-thicker, 4px) var(--borderWidth-thicker, 4px) 0; content: &quot;&quot;; display: inline-block; height: 0px; vertical-align: middle; width: 0px;"></span></summary><video src="https://user-images.githubusercontent.com/92835685/236143747-b5bef117-09c2-4b7e-b6b3-89e04eb505cc.mp4" data-canonical-src="https://user-images.githubusercontent.com/92835685/236143747-b5bef117-09c2-4b7e-b6b3-89e04eb505cc.mp4" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="box-sizing: border-box; display: block !important; border-top: var(--borderWidth-thin, 1px) solid var(--color-border-default)  !important; border-bottom-right-radius: var(--borderRadius-medium, 6px)  !important; border-bottom-left-radius: var(--borderRadius-medium, 6px)  !important; max-width: 100%; max-height: 640px; min-height: 200px;"></video></details>

## 🛠️Installation

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

## 🏃Getting Started

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

## 😮Acknowledgments

This project is based on the following repositories:

- [Whisper](https://github.com/openai/whisper#setup)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grapness](https://github.com/rhett-chen/graspness_implementation)
